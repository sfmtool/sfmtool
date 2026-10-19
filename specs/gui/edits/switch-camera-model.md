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
camera model` task), [../mcp-server.md](../mcp-server.md) (the tools),
[../point-cloud-rendering.md](../point-cloud-rendering.md) and
[../patch-rendering.md](../patch-rendering.md) (the preview buffers),
[../viewport-hud.md](../viewport-hud.md) (the proposal banner), and
[../viewport-navigation.md](../viewport-navigation.md) § "Maintain Z-up" (the
step function the preview's motion shares).

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
Before Apply, the **preview** shows that move: Preview retriangulation carries
the points and surfels of the camera's tracks smoothly in the 3D view to where
Apply will put them, so which parts of the cloud move, in which direction and
how far is visible in the second the move takes, in a way two static clouds
side by side do not show.

**The Kerry Park lenses** are the case this is built for.
`kerry_park_ground_truth_candidate_tk107.sfmr` has two `OPENCV_FISHEYE`
cameras, one per rig sensor, 480 × 480 with the principal point held at
(240, 240). The image content fills a circle of radius about 245 px, which at
f ≈ 129.6 is about 108° off the axis, while the polynomials' inverse blends
toward the identity ray from about 84° to 86° and cam0's forward map folds at
101.6°. So the ring of real image from 230 px to 245 px has no correct ray
under the current model: the inverse still returns one, the one its blend
toward the identity gives, wrong by the difference between the identity and the
lens, and a peripheral track was placed through it. `sfm xform --camera-model
SFMTOOL_FISHEYE,coeffs=8` fits the radial profile of both cameras to about a
hundredth of a pixel (rms 0.130 px and 0.384 px over θ ≤ 84.5° and 83.8°); the
rest is the difference between fx and fy (aspect 0.9978 and 1.0067), which a
single-focal model cannot represent. Whether that loss is acceptable for a
given lens is the reviewer's judgment, made in the viewer.

The switch itself moves pixels by under a pixel where there are observations.
What it gives is a model that can be refined out to the image circle: bundle
adjustment with the camera's spline released ([bundle-adjust.md](bundle-adjust.md)),
then Add Image to Tracks on the images ([add-image-to-tracks.md](add-image-to-tracks.md)),
now that tracks project past 86° to the right pixel, then bundle adjustment
again with observations where the spline had none. The observation
comparison's count past the old trusted bound shows whether the second step
reached the periphery, and whether the richer connectivity makes a better
reconstruction is judged by inspection, not by the residual metrics alone.

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
- The **observation comparison** the proposal's job sends when its switch
  returns (§ "The proposal's job"): the median, 90th percentile and largest
  error of the camera's observations under each camera over one fixed set, and
  over those past the current camera's trusted bound, with their count
  ([../camera-intrinsics.md](../camera-intrinsics.md) § "The proposal strip").
- **Preview retriangulation**, a checkbox, off when a proposal opens
  (§ "The preview").
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

### The proposal's job

While a proposal is open a **job** runs off the GUI thread
([camera_model_proposal/job.rs](../../../crates/sfm-explorer/src/camera_model_proposal/job.rs)).
It is not a background task ([../background-tasks.md](../background-tasks.md)):
those run one at a time in each of two queues and lock their node, and a
proposal pushes nothing, so it has nothing to lock, and its job must not hold off a
bundle adjustment on another node or be held off by one. It locks nothing,
pushes no version, writes no Action Log row and is not shown in the Background
Tasks panel. It follows the bench's live evaluation
([../bench.md](../bench.md) § "Live evaluation") instead:

- **Its inputs** are the proposal's generation, the document half of the
  version the proposal opened on, and whether the retriangulation is asked for.
  Nobody asks for a job: once per frame (`drive_camera_model_proposal_job`)
  the viewer finds a proposal whose inputs have no answer, and starts one. It
  waits while the node is busy.
- **One job runs at a time.** When the proposal's generation or version moves
  on while one runs, it is cancelled through its `Progress`, and the next
  starts once the cancelled one has reported back. Clearing the preview lets a
  running job finish, so ticking it again finds the answer. What a job sends is kept only when it is for the
  proposal's current generation, so an answer for an older generation is
  discarded rather than moved to.

One job runs, in order:

1. **The switch**, `switch_camera_model` over the value the proposal's fits
   read (the version's base, or its materialisation made once when the proposal
   opened), for the proposal's request, with
   `SwitchCameraModelOptions::outermost_keypoint` off, since the proposal read
   the keypoint when it opened. Its report carries the observation comparison,
   which the job sends to the panel as soon as the switch returns, waking the
   window. It takes no `Progress`, so a cancel is read when it returns. On a
   value with the inline keypoint column it is a fit and a pass over the
   camera's observations, which is milliseconds; on a `sift_files` value
   without it, it also reads the observations' pixels from the `.sift` files,
   on the job's thread, each file once. This step runs whether or not the
   preview is ticked or greyed. Where a file cannot be read the switch refuses
   (`SwitchCameraModelError::Observations`), and its sentence stands in the
   panel in place of the comparison.
2. **The retriangulation** of the proposal's tracks over the switched value,
   through `RetriangulateWhich::TheseInBulk` with the default options, only
   while Preview retriangulation is ticked or the wire asked for the preview,
   and only on a node that can be retriangulated.

The two steps are `switch_and_retriangulate`
([state/edits/switch_camera_model.rs](../../../crates/sfm-explorer/src/state/edits/switch_camera_model.rs)),
the computation the `Switch camera model` task runs, so the answer is the
version, label and Action Log sentence the task would push. The job then builds
the preview's targets and counts from the retriangulation's report on its own
thread, so the GUI thread only uploads them. The answer is kept by generation:
unticking and ticking again without a change of a control reuses it rather than
solving again, and the answer for the current generation holds its value,
since it is the version Apply installs (§ "Ending a proposal").

**Cost.** The materialisation is made once per proposal, and is the base
itself, shared by `Arc`, when the version's overlay is empty. Each job that runs
the retriangulation makes two whole-value copies: the switched value
`switch_camera_model` builds, and the answer value the retriangulation builds; a
job for the comparison alone makes the first. The answer for the current
generation is held while the proposal is open, so a proposal with the preview on
holds one extra copy of the value. The size factors cost one pass over the
images per job, not one per point: the camera centroid is taken once.

### The preview

The switch changes how rays map to keypoints, and Apply then retriangulates
the proposal's tracks through the new model, so the 3D effect of a proposal is
where those tracks are solved again, which is also where a bundle adjustment
after Apply starts from. Ticking **Preview retriangulation** moves the points
and surfels of the proposal's tracks, in the 3D view, smoothly from where they
are to where Apply will put them; clearing it moves them back the same way
([camera_model_proposal/preview.rs](../../../crates/sfm-explorer/src/camera_model_proposal/preview.rs)).
The checkbox is greyed, with `retriangulate_points`' refusal sentence as its
hover text, on a `sift_files` node without the inline keypoint column, whose
observations carry no pixel; the job still runs its switch there.

**The proposal's tracks** are every track with at least one observation in an
image taken through the proposal's camera, the points Apply retriangulates. A
track only other cameras observe reads the same rays under both models, is not
retriangulated, and does not move.

**What each point does** is read off the proposed run's outcome, looked up by
index, against the point's stored geometry. The run reads the value the
proposal's fits read, so the report's `index` is a row of it, and that value's
row map takes each row back to the point the 3D view draws: a row of the base's
point buffer for a base point and a row of the additions buffer for an added
one; a deleted point has no row. Every case compares the proposed answer with
what the value stores, which is what Apply replaces, so no run through the
current camera is needed.

- **Moves**: the answer is a position for a finite point, or a direction for a
  point stored as one. The target is the answer, which is what Apply writes. A
  finite point is drawn on the straight line from its stored position to the
  answer. A point at infinity is drawn on the same line between its stored
  bearing and the proposed one, normalised, so it turns along the shorter arc.
  A ranged point keeps its distance from its reference image's camera centre at
  both ends; between them the straight line puts it slightly inside that
  sphere. The motion starts at the stored position, so ticking the box causes
  no jump.
- **Loses its position**: the point is stored finite and the answer is a
  direction (a `Solved` outcome under `Marked`, `Thin`, `NoDepth`, `Behind`,
  `OverBar`, or `Ranged` at an infinite distance; under the default options,
  with the floor, the likelihood rule and the reprojection bar off, that is
  `Behind`). Apply writes it as that direction. A point cannot travel to
  infinity along a line, so the preview leaves it where it stands and moves its
  colour to the **failure colour**, `FAILURE_RGB` (`[235, 30, 30]`, beside
  `TINT_PALETTE` in `scene.rs`), as the motion runs; when the version lands it
  is drawn where the 3D view draws any point at infinity. These are the points
  the reviewer most needs to find: a track the proposed model's rays put behind
  a camera that sees it is the plainest sign that the proposed model is wrong
  somewhere, so the preview makes them stand out rather than recede. The
  failure colour is not in the tint palette, whose nearest entry is Vermillion,
  and it replaces the node's tint for these points. The banner gives their
  count by reason, and the reason for the selected point, in the words the
  Retriangulate edits' Action Log row uses
  ([../viewport-hud.md](../viewport-hud.md) § "The proposal banner").
- **Left where it was**: the outcome is `Kept`, fewer than two of the point's
  observations giving a usable ray. Apply keeps its stored geometry, and the
  preview neither moves nor colours it. A track with fewer than two
  observations is kept under any model, so the banner counts these apart.
- **Held**: never read by the run, unchanged by Apply, and does not move.
- **Placed**: a point stored as a direction whose answer is a position (a
  `Ranged` point at a finite distance). Apply writes the position and the
  banner counts it as moving, but a direction cannot travel to a place along a
  line, so the preview leaves it where it is drawn.

The answers usually differ a little from the stored points even where the
model hardly changes, because the stored points come out of a bundle
adjustment and a fresh triangulation does not reproduce it. That motion is part
of what Apply does, and the preview shows it with the rest. On Kerry Park a
peripheral track was placed through a wrong ray (the `OPENCV_FISHEYE` inverse's
blend toward the identity past its trusted range), so its motion is the
correction of that ray and can be large. A track that moves far is drawn like
every other point, with no cap, fade or separate list: a far move is either
such a correction or a sign that the proposed model is wrong, and in both cases
it is what Apply does.

**Surfels** move with their points and keep their orientation. A surfel's size
changes by `patch_frame_factor` for the move from the stored position to the
target, measured from the switched value's camera centroid taken once, which is
the factor the retriangulation rescales the patch frame by, so it keeps its
angular size as it moves and ends at the size Apply writes. A point at infinity
keeps its size. A surfel whose point loses its position takes the failure
colour with it ([../patch-rendering.md](../patch-rendering.md) § "The camera
model proposal's preview").

### The motion

A blend `s` runs from 0 (stored) to 1 (target), and each moving point is drawn
at `stored + s · (target − stored)`, normalised for a bearing. A point that
loses its position is drawn at its stored position, at full opacity, in
`mix(colour, failure colour, s)`, so the red builds up while the other points
travel and recedes when the box is cleared.

`s` is stepped one frame at a time by `motion::step`, the controller Maintain
Z-up turns `world_up` with
([../viewport-navigation.md](../viewport-navigation.md) § "Maintain Z-up"),
toward a goal of 0 or 1, with a cap of 2 per second and an acceleration of 10
per second², so a full move takes 0.7 s with 0.2 s ramps at each end. Clearing
the box while the motion runs sets the goal to 0, and the signed velocity
slows the points and sends them back without a jump. The frame asks for
another while the blend moves.

**A new answer while the box is ticked** comes after a control changed, and
gives the moving points new targets. Each point's drawn position is `stored +
s · (target − stored)`, so swapping the targets at any `s` above 0 would make
the points jump. The new answer is therefore held as pending and the goal set
to 0: the points run back to their stored positions on the old targets, the
preview buffers are rewritten with the new targets when `s` reaches 0, and the
goal goes back to 1. A newer answer arriving during the run back replaces the
pending one. When `s` is already 0 the targets are written at once, and while
the job has no answer for the current generation the goal is 0.

### What the preview does not change

The value: every panel other than the 3D view reads the stored points, Track
View and Image Detail's reprojections among them, and so does the CPU side of
the 3D view: Go to Point frames a point at its stored position, and the
bench's figures read stored positions. The one CPU-drawn part of the 3D view
that would visibly disagree with a moving point is its track rays, which end at
the point, so while `s` is above 0 the selected point's track rays are not
drawn when the preview moves that point, and they return when `s` is back at 0.
A point that loses its position stands where its rays end, so its rays stay
drawn.

The GPU side follows the drawn position. The pick pass is the same vertex
shader, so a click on a moving point picks it where it is drawn and selects the
point it is, and Alt+click reads the orbit target's depth from the drawn
position. The camera frustums and the background mesh stay on the current
model, because they would change by fractions of a pixel, which are not
visible at viewport scale.

The motion is not exaggerated. Each point that moves ends at the position
Apply writes for it, however small the move, so what the preview shows at its
end is the version Apply pushes, except that a point that loses its position
stands where it was, in the failure colour, rather than at infinity.

The preview's buffers are created when the first answer is shown on the node
and dropped when the proposal ends; the node's own point and surfel buffers are
never written, so ending the preview restores nothing
([../point-cloud-rendering.md](../point-cloud-rendering.md) § "The camera
model proposal's preview").

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

Where the proposal's job already holds its answer for the current generation
(§ "The proposal's job"), that answer is the version, and Apply installs it on
the GUI thread without solving again: the same version, label and Action Log
row the task below would push, timed from the click. Only a busy node refuses
it, since nothing runs in the background.

Otherwise it runs as the **`Switch camera model`** background task
([../background-tasks.md](../background-tasks.md)), which locks the node,
reports the switch and the retriangulation's stages, is cancellable on either
side of the switch and inside the retriangulation, and pushes the version when
it finishes; the proposal ends when the task starts. A refusal to start
(another task running in the viewer) leaves the proposal open. The implicit
endings below apply by the same rule.

**Every ending ends the preview in the same frame**: the preview buffers are
dropped and the node's points are drawn at the positions its value holds.
Clearing the checkbox is the way to see the motion back. Apply does not wait
for the motion: the version it pushes holds the retriangulated positions, and
the new base it installs recreates the node's point buffer from them. A preview
that had run to its end hands over to the new base with no visible change for
the points that move, since their targets are the positions Apply writes, to
the last bit of the `f32` the 3D view draws, and each surfel's half-vectors at
the end are the ones Apply writes; a point that loses its position goes from
its stored position, in the failure colour, to where the 3D view draws a point
at infinity. What can still change on screen is what the 3D view derives from a
new base: the node's splat size is measured again from the new positions
(`upload_points`, [../point-cloud-rendering.md](../point-cloud-rendering.md)
§ "Point Sizing"), and the eye-dome lighting with it, so in a dense part of
the cloud splats grow or shrink by the change in point spacing (about 2 % on
Kerry Park's camera 0) while every point stays where it was. Without the preview the
camera's tracks move in one frame when the version lands. Where Apply runs as
the task, the preview ends when the task starts and the points stand at their
stored positions until it lands.

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
is refused, for instance because the `.sift` files the switch reads its pixels
from are gone, its refusal is the failed row and the proposal is dropped with
a `View` row after it, `Dropped the camera model proposal for camera 0 of
kerry_park: its Apply was refused`. An Apply while the slow queue runs a task
on another node is not refused: the `Switch camera model` task waits behind it,
and the proposal's node is busy with the waiting sentence from then on
([../background-tasks.md](../background-tasks.md) § "Two queues"). A close
applies first, before the close is asked about; where that starts the background task the node is busy, and the
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
  proposal with its own `View` row, and one behind a busy slow queue waiting to
  apply and then landing; a refused fit shown and not applied;
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
  proposal is open and `distortion` after it is cancelled; and the preview on
  the wire: `propose_camera_model` with `preview_retriangulation` answering
  with a `pending` triangulation block and keeping the generation,
  `get_camera_intrinsics` reporting `ready` with the counts and the
  observation comparison once the job lands, `animate: false` leaving `s` at 0
  until the answer lands and setting it to 1 in that frame, a clear and a
  tick under it replying with `s` already at 0 and 1, and the proposal
  form with the answer in hand replying with the version at once.
- `camera_model_proposal/job/tests.rs`: without the preview the job sending
  the comparison alone, pushing nothing and writing no row; Apply with the
  answer in hand installing it with no background task, and pushing the
  cameras, positions, label and row the task pushes for the same proposal; an
  answer for an older generation not installed, Apply then running the task; a
  running job for an older generation cancelled and its answer not kept; on a
  `sift_files` node the preview refused with `retriangulate_points`' sentence
  while the switch's comparison still arrives, and the switch's refusal in its
  place once the `.sift` files are gone; Cancel, Apply and an implicit ending
  each dropping the preview, and hiding the node leaving it; and on a version
  with a deleted point and an addition, the targets laid out by the base and
  additions buffers, the deleted row not moving, the added point moving, and
  each moving base row's target where the version Apply pushes holds that
  point; and for every point of a node with patch frames, a moving row's target
  equal to the installed position as an `f32`, its half-vectors times its size
  factor equal to the installed ones, a still row's point and frame unchanged,
  and a lost point installed as a direction.
- `camera_model_proposal/preview/tests.rs`: every outcome against each kind of
  stored point (held, kept, a finite move with its surfel factor, a ranged and
  a pruned one, a finite point lost to `Behind`, a point at infinity turning to
  a unit bearing at its size, a direction placed, an unchanged answer); the
  targets and counts read off a report over a value with a deleted point (rows
  taken back through the row map, the deleted row still, the lost counted by
  reason largest first, the median and 90th percentile of the shifts); the
  blend run up in about 0.7 s and back; a new answer at `s = 1` swapped in only
  at 0 with no jump in `s`; no answer for the current generation running it
  down and keeping the old targets; and `animate: false` jumping to its goal.
- `motion.rs`: the shared step reaching its goal without overshoot, a full
  preview move taking about 0.7 s, and a goal flipped mid-motion reversing
  with no jump in the value or its speed; `viewer_3d/righting.rs` keeps
  Maintain Z-up's timings.
- `scene_renderer/upload/tests.rs`, on the `noop` backend: the preview buffers
  created at the base and additions buffers' row counts with `COPY_DST`, the
  surfel rows written by slot through `slot_of_point` with a still row's flag
  0 and factor 1, an addition's surfel row found past the base's rows, `s` in
  the node's `ReconUniforms`, and the buffers dropped when the preview ends,
  when its rows no longer fit and when a new base arrives.
  `scene_renderer/pipelines/tests.rs` builds both pipeline variants and checks
  the three flags and the failure colour against `common.wgsl`.
- `camera_model_proposal/banner.rs`: the counts in the banner's words and the
  lost points in the failure colour; and at 341, 482 and 1800 px the banner's
  frame and every line inside the view with no line over another (the test
  fails with the wrapping removed); every character the banner can show laid
  out in a font that has it, the arrow in the monospace font; and the banners
  starting below the status line.

---

## Non-goals

- Refitting several cameras at once. A rig's cameras are proposed and applied
  one by one, each its own version.
- Typing a camera's parameters. A lens changes through a fit or a solve.
- Freeing the principal point in the fit (§ "The strip").

---

## Open questions

- **The dropped aspect.** `SFMTOOL_FISHEYE` has one focal. The Kerry Park fx/fy
  differences of 0.2 % and 0.7 % are most likely fitting noise on square
  pixels, and the switch drops them visibly. If a real lens needs them, an
  aspect parameter is a format change to the spline models. A Kerry evaluation
  run would settle it: switch tk107, Add Image to Tracks over all images,
  bundle adjust with the spline released, then report the observation counts
  by θ band before and after, the fitted spline against the image circle at
  245 px, and whether the residuals show an azimuthal pattern. The first
  release on tk107 (`sfm xform --camera-model SFMTOOL_FISHEYE,coeffs=8
  --bundle-adjust`) brought the median residual from 0.298 px to 0.274 px and
  moved the coefficient whose support starts near 86° from 0.09 to −0.48 on
  cam0 and from 0.25 to −0.43 on cam1, with at most 58 and 39 observations past
  the old trusted bound reaching it, some off by hundreds of pixels under either
  model; whether that periphery describes the lens or those observations is
  judged in the viewer.
- **Whether two whole-value copies per job is fast enough** (§ "The proposal's
  job"). If not, on a large node, `retriangulate_points` could solve through a
  camera table the caller states and return its statuses and positions without
  building a value, as a mode of the same function with the same rules. Apply
  would then solve again to build the version rather than install the job's
  answer, so the saving is in the jobs a proposal's changes start, not in
  Apply.
- **Whether other previews share the motion.**
  [`../../drafts/move-camera-preview-amendment.md`](../../drafts/move-camera-preview-amendment.md)
  previews a re-triangulation under Move Camera by writing over the base's
  rows, and could use the preview buffers and the blend instead.
