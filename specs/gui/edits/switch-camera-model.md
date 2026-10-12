# Switch Camera Model

An edit that replaces one camera's lens model with one fitted to it and
installs the answer as the node's next **version**, which an undo steps back
out of. In the window it is reached through a **camera model proposal**: the
camera fitted to the model chosen, shown in the Camera Intrinsics panel beside
the current camera before anything changes, and **Apply**, which pushes the
switch together with the retriangulation of the camera's tracks. A spline
camera switched to its own spline model with another coefficient count or
domain end is the **refit** of its spline, made from the same proposal. The MCP
tool `switch_camera_model` reaches the lens-only switch directly, and applies
an open proposal in its `proposal: true` form.

Related specs:
[../../core/reconstruction/switch-camera-model.md](../../core/reconstruction/switch-camera-model.md)
(the core function this wraps, `fit_camera_switch`, and the report),
[../../core/camera/refit-camera-intrinsics.md](../../core/camera/refit-camera-intrinsics.md)
(the lens fit, and `refit_spline`),
[../../core/reconstruction/triangulation-rules.md](../../core/reconstruction/triangulation-rules.md)
(`retriangulate_points` and `RetriangulateWhich::TheseInBulk`),
[../camera-intrinsics.md](../camera-intrinsics.md) (the panel that shows the
proposal), [retriangulate-point.md](retriangulate-point.md) (the
retriangulation edits, whose options and counts Apply shares),
[bundle-adjust.md](bundle-adjust.md) (the edit that then refines the new
coefficients), [../background-tasks.md](../background-tasks.md) (the `Switch
camera model` task), [../mcp-server.md](../mcp-server.md) (the tools), and
[../../drafts/switch-camera-model.md](../../drafts/switch-camera-model.md) (the
preview of the retriangulation in the 3D view, which is not built yet).

---

## Purpose

A camera model is the function that maps a ray leaving the camera to a pixel.
Switching a camera to another model replaces that function with one fitted to it
over the angles where the old one is trusted. The main use is moving a fisheye
camera from a COLMAP polynomial model, which fails a little past 90° off the
axis, to `SFMTOOL_FISHEYE`, whose spline and linear tail are defined out to
180°. A spline camera's coefficient count and domain end say how finely its lens
curve can bend and where the bending stops; bundle adjustment refines the
coefficients a camera has but cannot change how many there are or where the
domain ends, so changing them is a refit of that one camera, which a later
adjustment then fits to the observations.

The switch is a choice a reviewer makes by looking: whether the loss a fit
reports (a dropped aspect ratio, a monotone constraint that bound) is
acceptable for a given lens. So the window shows the proposed camera beside the
current one before it is applied. The outermost keypoint of the camera's images
is on screen beside the domain it should reach.

The points were triangulated through the old model's rays. After a lens-only
switch they stand where those rays put them, so the value holds a camera and
points that disagree. The window's Apply therefore moves the camera's tracks in
the same version: the value agrees with the lens it now holds, and the next
bundle adjustment starts from the points the new model gives. The wire's
direct form stays the lens-only switch `sfm xform --camera-model` makes.

---

## Invocation

Three entries in the window open a proposal on one camera:

- **Switch Model…** in the Camera Intrinsics panel's header, to the left of
  `Copy ▾` ([../camera-intrinsics.md](../camera-intrinsics.md) § "1. Header"),
  for the camera the panel shows. The panel reports the click as
  `IntrinsicsDetailResponse::switch_model`.
- **Switch Camera Model…** on the context menu of a row of the Scene tree's
  Camera Intrinsics group ([../scene-graph.md](../scene-graph.md)).
- **`Edit > Switch Camera Model…`**, for the selected camera, greyed when no
  camera or image is selected.

The last two bring the Camera Intrinsics panel to the front. Every entry ends a
Move Camera lock held on the node first, by the lock's own implicit rule: a
commit when the pose has moved and a silent drop otherwise
([move-camera.md](move-camera.md)). Opening a proposal selects its camera. A
rig's cameras are proposed and applied one at a time, each its own version.

Each entry is greyed, with the reason as its hover text, when
`AppState::switch_model_refusal` gives one:

- the node is busy with a background task, with the sentence the busy node's
  other edits carry;
- a proposal is already open on another camera of the node (*"A camera model
  proposal is open on camera 1 of kerry_park. Apply or cancel it first."*);
- the node has no posed image.

The panel's own button is also greyed while the proposal for its camera is open
below it.

---

## The proposal

While a proposal is open, the node has a **proposed camera**: a
`CameraModelProposal` in `AppState::camera_model_proposal`, in
[camera_model_proposal.rs](../../../crates/sfm-explorer/src/camera_model_proposal.rs).
It is viewer state, as the Move Camera lock's pending pose is, and is in no
version until it is applied. One proposal is open at a time. It holds the
camera as it was when it opened, the value the fits read (the version's base,
or its materialisation, made once, when the version carries point edits), the
outermost keypoint of the camera's images and the keypoints of the camera's
observations (the inline column, or the image's `.sift` file), and the inputs of
the current fit with the fit itself. The outermost keypoint and the keypoints
come from one pass over the camera's images on the GUI thread when the proposal
opens, which reads each image's `.sift` file once for both
(`observations::read_camera_keypoints`); the outermost keypoint is the one
`outermost_keypoints` gives over the version's base. It carries a
**generation**, a counter every change of the fit inputs advances; a change
that leaves them as they were keeps it. The generation is drawn from one
counter for the whole viewer, so no two proposals share one and a cache keyed
by it never serves one proposal's camera for another's.

### The strip

The panel shows a strip of controls under the header
([intrinsics_detail/proposal.rs](../../../crates/sfm-explorer/src/intrinsics_detail/proposal.rs)):

- **Model**: the targets the camera can be fitted to, with `SFMTOOL_FISHEYE`
  first for a fisheye camera and `SFMTOOL_PINHOLE` first otherwise. A spline
  camera's own model is in the list, labelled as the refit
  (`SFMTOOL_FISHEYE (refit spline)`), and the proposal opens on it for such a
  camera; any other camera's own model is left out, and the proposal opens on
  the list's first target. Choosing another target starts it at its own
  defaults.
- **Coefficients** (2–32, `SPLINE_COEFF_COUNT_RANGE`), for a spline target,
  starting at the count `RefitTarget::coeff_count_for` gives: the camera's own
  for a refit, and 8 otherwise. For a refit, `now 8` beside it.
- **Fit to θ (°)**, the largest incidence angle the fit samples. It shows the
  default and where it came from (`default 84.5° (trusted_bound)`), can be
  lowered, and a `Default` button puts it back. The default is
  `default_fit_angle`'s, read when the proposal opens, so it is shown beside a
  refused fit as well, which the refusal often names as the fit's largest
  angle; lowering it is how such a fit is made to succeed. For a camera the
  core finds no default for (one with no image) the row reads `no default
  angle` instead of a number. Not shown for a refit, since a given angle makes
  the core fit the camera like any other source.
- **Spline domain (°)**, for a spline target: the camera's own domain end for a
  refit, with `now 150.1°` beside it, and the far image corner the fit places
  otherwise; it can be edited up to 180°, and below 90° for `SFMTOOL_PINHOLE`.
  Under it, the outermost keypoint of the camera's images
  ([../../core/reconstruction/outermost-keypoint.md](../../core/reconstruction/outermost-keypoint.md)),
  `outermost keypoint: 230.3 px, 95.8° observed; 259.2 px, 108.8° detected`,
  and a **Use 108.8°** button that sets the domain to that angle: the detected
  keypoint's, and the observed one's where no `.sift` file could be read. The
  default domain is not changed by it: a circular fisheye is trimmed to its
  image circle by choice.
- The fit's **report** in one line, for example `rms 0.130 px, radial 0.013 px,
  max 0.292 px over θ ≤ 84.5°; aspect fy/fx 0.9978 dropped (single focal)`,
  with `; monotone constraint bound at 1 angle, 113.2°` added where the
  constraint bound, since over that range the proposed curve is the closest
  invertible one rather than the current one. A refused fit shows `Refused:`
  and the core's sentence instead, in the error colour.
- **Apply**, greyed while the fit is refused, and **Cancel**.

Every change refits at once, on the GUI thread, with
`sfmtool_core::reconstruction::switch_camera_model::fit_camera_switch`, the fit
the switch itself makes: a few milliseconds and one pass over the poses and
point positions, with no pixel read. So the camera the panel shows is the
camera Apply installs for the same request. The fit releases the focal length
and the distortion parameters and keeps the principal point fixed, as the lens
fit does; the strip offers no control that frees it, because no later
adjustment of a spline camera frees it either.

The rest of the panel compares the two cameras: the **Parameters** and
**Derived** tables show the current camera and the proposed one side by side,
and the proposed column's derived report is cached by the camera and the
proposal's generation, apart from the current camera's. The projection plot
draws both cameras against each other, with a third plot of the change and an
observation rug under the shared axis
([../camera-intrinsics.md](../camera-intrinsics.md) § "4. Projection plot").

### The Image Detail panel

While the proposal is open, the Image Detail intrinsics layer shows what it
does to the photographs of the camera: opening it selects the layer's **Change
to proposed** field mode, and its ending restores the mode before it. In that
mode the layer draws arrows from the current camera's pixel for a ray to the
proposed camera's, inside the current camera's trusted bound, the proposed
camera's rings every 5° past the fit's largest angle, and, for each observation
of the image, its keypoint with a line to its reprojection under each camera.
The axes and rings follow the proposed camera unless the layer's popup puts
them back on the current one
([../camera-intrinsics.md](../camera-intrinsics.md) § "The Change to proposed
field").

### A same-model target

`switch_camera_model` takes any target, the camera's own model included. For a
spline camera (`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`) its own model is the refit:
with no fit angle given, the core fits it with `refit_spline`, over the whole
new domain, with the domain end kept exactly where none is given, constrained
to stay monotone, and reporting `ThetaFitSource::SplineDomain`. The proposal
asks for exactly that: no model named, no fit angle, the count, and a domain
only when the field has moved from the camera's own (a field set back to it is
untouched), so an untouched field keeps the domain bit for bit. A proposal at
the camera's own count and domain can be applied: it is the closest monotone
curve to the current one, which differs from the camera where the constraint
binds.

Any other camera's own model is not in the Model list: the core fits the model
to itself over the fit angle, which changes the camera by no more than the
fit's error, and a polynomial model has no count or domain. Switch Model… is
not greyed for such a camera, since its other targets are offered. The wire's
`propose_camera_model` does take the camera's own model for any camera, and a
fit angle on a spline camera's own model; the panel shows a proposal the wire
opened as it was opened, its Model list holding the target named and Fit to θ
the angle given.

---

## Ending a proposal

**Apply** (`AppState::apply_camera_model_proposal`, in
[state/edits/switch_camera_model.rs](../../../crates/sfm-explorer/src/state/edits/switch_camera_model.rs))
ends the proposal and pushes one version: the direct switch below with the
proposal's request, followed by `retriangulate_points` over the proposal's
tracks on the switched value, through `RetriangulateWhich::TheseInBulk` and
with the default options the Retriangulate edits use
([retriangulate-point.md](retriangulate-point.md)). The **proposal's tracks**
are every point with at least one observation in an image taken through the
camera; a track only other cameras observe reads the same rays under both
models and is left alone. The edit composes the two core calls and computes
nothing of its own.

It runs as the **`Switch camera model`** background task
([../background-tasks.md](../background-tasks.md)), which locks the node,
reports the switch and the retriangulation's stages, is cancellable on either
side of the switch and inside the retriangulation, and pushes the version when
it finishes; the proposal ends when the task starts. A refusal to start
(another task running in the viewer) leaves the proposal open.

On a `sift_files` node without the inline keypoint column,
`retriangulate_points` refuses, since its observations carry no pixel. Apply
there pushes the switch alone: the direct switch's version, label and row, on
the GUI thread, with `; its tracks were not retriangulated: ` and
`retriangulate_points`' refusal sentence added to the row. Where the switch
itself refuses because a `.sift` file cannot be read
(`SwitchCameraModelError::Observations`), Apply is refused with that sentence
and the proposal stays open.

**Cancel**, and `Escape` while the pointer is over the panel showing the
proposal, drop it with a `View` row. `Escape` cancels only when no popup was
open and no field was being edited, so the `Escape` that closes the Model
dropdown or ends the editing of a field does only that
([../camera-intrinsics.md](../camera-intrinsics.md) § "The proposal strip"). The
row reads `Cancelled the camera model proposal for
camera 0 of kerry_park`.

**The implicit endings** follow the Move Camera lock's. Selecting another node,
closing the proposal's node (by its context menu, Close All, or the wire), and
an MCP edit tool naming the node apply the proposal when its fit succeeded,
since an edit has a history and undo reverses it, and drop it, with a `View`
row, when the fit was refused (`…: its fit was refused`). Where the Apply itself
is refused, for instance because a task is running on another node, its
refusal is the failed row and the proposal is dropped with a `View` row after
it, `Dropped the camera model proposal for camera 0 of kerry_park: its Apply
was refused`. A close applies first, before the close is
asked about; where that starts the background task the node is busy, and the
close is refused with the busy sentence, as any close of a busy node is. An MCP
edit tool that set off the task is refused the same way. The frame checks the
selection once per frame (`AppState::tend_camera_model_proposal`), after the
MCP drain.

Three other states end or block a proposal:

- **Undo, redo or a history jump on the node**, from the menus or the wire,
  drop it, writing `Dropped the camera model proposal for camera 0 of
  kerry_park: the history moved off the version it was fitted to` as a `View`
  row before the move's own. The proposal was fitted to the camera of the
  version it opened on; applying it first would put a version on the history
  that the undo then steps back out of. A value that changes under the
  proposal any other way (a save that folds in the overlay, say) drops it too,
  at the next frame. A bench step, which shares the version's document half,
  leaves it open.
- **The node's other edits** are greyed while a proposal is open, with *"Apply
  or cancel the camera model proposal first."* as the hover text
  (`AppState::edit_refusal`): the Scene tree's whole-node edits, the image
  menu's four entries, the 3D viewport's point and patch menus, the Edit menu's
  Delete Point and Delete Image, and Move Camera. An edit reached another way
  in the window, such as Track View's commit of a bench track, is refused with
  the same sentence, and a background task other than Apply's cannot start on
  the node.
- **Hiding the node** leaves the proposal as it is.

---

## The direct switch

`AppState::switch_camera_model` takes the request the wire's direct form and
the proposal both make:

| Field | Default |
|-------|---------|
| `camera` | required: the camera's index in the node's table |
| `camera_model` | the camera's own model |
| `coeff_count` | the camera's own count when the target is its own spline model, else the core default, 8 |
| `spline_domain_deg` | the camera's own domain end in a refit, else the far image corner |
| `theta_fit_deg` | the core default; given, even a refit is an ordinary fit over that angle |

It runs **on the GUI thread**: a fit is a few milliseconds, and the only files
read are `.sift` positions, on a `sift_files` node without the inline keypoint
column, for the observations' pixels the report's comparison and the points'
stored errors are measured against. It shows no outermost keypoint (the
proposal reads its own when it opens), so it asks the core switch to leave that
out of the report and reads no file for it.

### The version

A bulk edit. The value is the whole new base the core function produced, with
an empty overlay; the current value is materialised first when its overlay is
not empty. The direct switch moves no pose, point, keypoint or track and
deletes nothing, so the `RowMap::by_scan` read off its input and output is the
identity, and the selection stays where it is. Apply's retriangulation rewrites
the proposal's tracks in place, at their own rows, so its map renumbers nothing
either. The camera table changes and the stored errors of the points the
camera's images observe are recomputed, so the panels drop what they cached
about the node, the Camera Intrinsics panel's derived report and the Image
Detail overlay among it.

The version's label is, for a refit of a spline,

`Refit spline of camera 0 of kerry_park: 8 → 12 coefficients, domain 150.1° → 108.8°`

with a part that did not move named as kept (`8 coefficients kept`, `domain
150.1° kept`), and, for a change of model,

`Switched camera 0 of kerry_park from OPENCV_FISHEYE to SFMTOOL_FISHEYE`.

Apply's version adds ` and retriangulated its tracks` to either.

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
observations with an error under both models. Apply's row follows the fit's
report with `; ` and the counts the Retriangulate edits write for a whole-value
run (`3020 of 3020 points moved, …, median shift 0.0123`), read off the
retriangulation's report.

A refusal is one **failed** entry, `Switch camera model of <node> refused:
camera 0: <reason>`, carrying the core function's sentence: a camera the table
does not have, a count on a model without a spline, a domain end the model
cannot have. Nothing is pushed. A perspective target refused because a point
lies 90° or more off the camera's axis names that point, by the index the
version on screen gives it, and says the angle is the ray to the point's
position rather than a keypoint read through the lens, since that angle can
differ widely from the outermost keypoint the proposal shows
([core](../../core/reconstruction/switch-camera-model.md)). The proposal's
fit line shows the same sentence.

---

## Testing

Core: [../../core/reconstruction/switch-camera-model.md](../../core/reconstruction/switch-camera-model.md)
§ "Testing", which pins the refit of a spline to `refit_spline` and
`fit_camera_switch` to the switch's own fit.

Explorer (`sfm-explorer` lib tests, headless):

- `camera_model_proposal/tests.rs`: every listed target is one the switch
  takes; the Model list's order and the own model left out or kept; a spline
  camera's proposal opening as the refit at its own count and domain with no
  fit angle, its request naming no model and no fit angle, and a domain set
  back to the camera's own read as untouched; a polynomial camera opening on
  the first target with the default fit angle; the generation advancing only
  when the inputs change; the outermost keypoint's button taking the detected
  angle, the observed one without a detected and nothing without either, and
  the keypoint text labelled by its source; Apply starting the `Switch camera
  model` task, which pushes one version labelled with the retriangulation,
  holding the switched camera and moving only the camera's tracks, undone back
  to both; Apply on a `sift_files` node being the switch alone with the refusal
  sentence in its row, and refused with the proposal left open once the `.sift`
  files are gone; Cancel; an implicit ending whose Apply is refused dropping the
  proposal with its own `View` row; a refused fit shown and not applied;
  selecting another node applying the proposal and dropping a refused one;
  closing applying first and the close then refused while busy; undo, redo and a
  jump each dropping it with one `View` row; a value changed under it
  dropping it, and a bench step leaving it while the bench's commit is refused;
  hiding the node leaving it; the node's other edits, its image
  menu, a background task and Move Camera refused while it is open; Switch
  Model… greyed on a busy node, on another camera, and on a node with no posed
  image; and opening a proposal ending a held Move Camera lock. On a fisheye
  fixture: the rug holding one angle per observation, ascending, with the wide
  observations past the trusted bound; an image's observations reprojected
  under both cameras, the current one onto its keypoint, and a point at infinity
  reprojected as its direction; the Change to proposed field mode selected on
  opening and restored after Cancel and after Apply, a mode chosen while open
  standing until the end; a reopened proposal taking a new generation; and the
  one pass over the images giving the outermost keypoint `outermost_keypoints`
  gives, on a `sift_files` node and an inline one, while reading each `.sift`
  file once.
- `intrinsics_detail/tests.rs`: the header's `Switch Model…` and no `Refit
  spline…`; a refit proposal's strip and the two cameras side by side; a change
  of model's Fit to θ with its default; Apply, Cancel and `Escape` reported;
  the proposed report cached apart from the current one; the header and the
  two tables painting no string over another at narrow and wide widths, the
  tables stacked when narrow and side by side when wide; an `Escape` that
  closes the Model dropdown, or ends the editing of Fit to θ, not cancelling
  and the next one cancelling; the `Default` button and the default's label
  inside the panel at the stock window's 341 px, with nothing past its edge and
  no header line starting with a `·`; and a
  refused proposal showing its default fit angle and source, keeping a
  wire-given value the field's range does not hold, and the current tables
  alone.
- `scene_graph/tests.rs`: a camera row's menu offering `Switch Camera Model…`,
  greyed on another camera while a proposal is open, with the node's
  retriangulation greyed too.
- `state/edits/tests.rs`: a refit pushing one version with the label naming
  both changes, only the named camera changing, the log row carrying the fit's
  numbers, and undo bringing the old camera back; a domain end past 90° on
  `SFMTOOL_PINHOLE` and a count on a `SIMPLE_PINHOLE` refused naming the
  camera, pushing nothing.
- `background/tests.rs`: `Switch camera model` stopping when cancelled.
- `mcp/tests/edit.rs`: `switch_camera_model` parsing in both forms and refusing
  a fit argument beside `proposal: true`; a refit with the camera's own count
  and domain kept by default, a named domain keeping the count, a named model
  switching models, and a missing camera and an impossible domain refused;
  `propose_camera_model` pushing nothing, keeping its generation for the same
  inputs and advancing it for new ones, reported by `get_camera_intrinsics`,
  and dropped by `cancel_camera_model_proposal`; an unknown model refused and a
  refused fit carried; the proposal form deferring to the task and moving the
  tracks with the direct form's `fit` object in the reply, undone over the wire;
  on a `sift_files` node the switch alone answered at once with `fit`; a direct
  switch applying an open proposal first and then refused while busy, and
  moving no point on its own; an undo over the wire dropping the proposal; and
  `get_image_detail_display` reporting the `change_to_proposed` field while a
  proposal is open and `distortion` after it is cancelled.

---

## Non-goals

- Refitting several cameras at once. A rig's cameras are proposed and applied
  one by one, each its own version.
- Typing a camera's parameters. A lens changes through a fit or a solve.
- Freeing the principal point in the fit (§ "The strip").
