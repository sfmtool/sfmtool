# Switching a camera to another camera model in the viewer

**Status:** Draft

Decided in outline:

- The switch is a fit, one core operation, reached from the Camera Intrinsics
  panel, from MCP and from `sfm xform --camera-model`. The core operation, the
  CLI and the direct switch are built: the lens fit is
  [`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md),
  the reconstruction-level switch is
  [`../core/reconstruction/switch-camera-model.md`](../core/reconstruction/switch-camera-model.md),
  the CLI is
  [`../cli/reconstruction/xform/xform-command.md`](../cli/reconstruction/xform/xform-command.md)
  § "Camera Model", and the viewer's direct switch is
  [`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md):
  the "Refit spline…" action in the Camera Intrinsics panel header, which
  gives a spline camera a new coefficient count or domain, and the MCP tool
  `switch_camera_model`, which applies a switch or a refit at once.
- In the viewer, a change of model is shown as a proposal for one camera, drawn
  against the current model, before it is applied. Apply is the shipped direct
  switch, and pushes one version. That is what remains, with the MCP tools
  `propose_camera_model` and `cancel_camera_model_proposal` and the proposal
  form of `switch_camera_model`.
- While a proposal is open, a **Preview retriangulation** checkbox moves the
  points and surfels of the proposal's tracks smoothly, in the 3D view, to
  where the proposed model would triangulate them, and back when it is cleared.
  Apply does not move them; the switch leaves points alone.

Not decided: see [Open questions](#open-questions).

Builds on: `retriangulate_points`
([`../core/reconstruction/triangulation-rules.md`](../core/reconstruction/triangulation-rules.md)
§ "The report"), whose report carries a `RetriangulatedPoint` for every point
the call was asked about: its `index` in the value passed in, its `new_index`
in the value returned, consistent with the call's `PointMap`, and a
`RetriangulateOutcome`. The outcome is `Held`, `Kept` (fewer than two usable
rays, so the stored geometry stays), or `Solved` with the `PointVerdict` that
decided it. This draft reads gains and losses of a position off those outcomes
and adds no verdict logic of its own.

Amends:

- [`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md), whose header
  gains the Switch Model… button beside "Refit spline…", and whose § "Scene
  Graph: the Camera Intrinsics group" gains the rows' context menu.
- [`../gui/scene-graph.md`](../gui/scene-graph.md), whose Scene tree gains a
  context menu on the rows of the Camera Intrinsics group, which have none
  today.
- [`../gui/mcp-server.md`](../gui/mcp-server.md), which gains
  `propose_camera_model` and `cancel_camera_model_proposal`, whose
  `switch_camera_model` gains the proposal form, and whose
  `get_camera_intrinsics` gains the `proposed` block.
- [`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md),
  the direct switch the proposal's Apply pushes. Its § "Invocation" gains the
  proposal's three entries, and its non-goal "A dialog for a change of model
  family" is replaced by the proposal. Its non-goal "Refitting several cameras
  at once" stands: a proposal is for one camera and Apply pushes one version
  with the label the edit already writes.
- [`../core/reconstruction/switch-camera-model.md`](../core/reconstruction/switch-camera-model.md),
  whose non-goals name the proposal this draft describes, and which gains an
  option that leaves the outermost keypoint out of the report, so a caller that
  does not need it reads no `.sift` file.
- [`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
  § "The frame follows the depth", whose placement distance gains a form that
  takes a camera-cloud centroid computed once, so a whole-value write-back
  costs one pass over the images rather than one per point, and the preview
  reads its surfel size factor from the same rule.
- [`../gui/point-cloud-rendering.md`](../gui/point-cloud-rendering.md), whose
  base and additions point buffers gain a preview buffer, with a target per
  row blended in the shader.
- [`../gui/patch-rendering.md`](../gui/patch-rendering.md), whose surfels gain
  a target centre, a size factor and the failure colour, written by slot.
- [`../gui/viewport-hud.md`](../gui/viewport-hud.md) § "The lock banner borrows
  the style", which gains the proposal banner.
- [`../gui/viewport-navigation.md`](../gui/viewport-navigation.md) § "Maintain
  Z-up", whose `righting::step` is generalised to a one-dimensional controller
  that the preview's blend also calls.

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
circle of radius about 245 px, which at f ≈ 129.6 is about 108° off the axis.
The polynomials' inverse blends toward the identity ray from about 84° to 86°,
and cam0's forward map folds at 101.6°, so the ring of real image from 230 px
to 245 px has no correct ray under cam0. The inverse still returns a ray there,
the one its blend toward the identity gives, and that ray is wrong by the
difference between the identity and the lens.

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
and 39 observations past the old trusted bound reach that coefficient, and some
of them are off by hundreds of pixels under either model. The two coefficients
no observation reaches came back unchanged. Whether the periphery that fit
produces describes the lens or those observations is judged in the viewer.

## The viewer

### Invocation

- A **Switch Model…** button in the Camera Intrinsics panel header, beside
  "Refit spline…" and `Copy ▾`.
- A **Switch Camera Model…** entry on a new context menu on the rows of the
  Scene tree's Camera Intrinsics group. The rows have no context menu today.
- `Edit > Switch Camera Model…`, greyed when no camera is selected.

Each opens the proposal for that one camera. A rig's cameras are proposed and
applied one at a time, each its own version, as the direct switch already does.

Each is greyed, with the reason as its hover text, in three cases:

- the node is busy with a background task, with the sentence the busy node's
  other edits carry ([`../gui/background-tasks.md`](../gui/background-tasks.md)
  § "What the rest of the viewer does meanwhile");
- a proposal is already open on another camera of the node;
- the node has no posed image.

Opening a proposal on a node where the Move Camera lock is held ends the lock
first, by the lock's own implicit rule: a commit when the pose has moved, and
a silent drop otherwise
([`../gui/edits/move-camera.md`](../gui/edits/move-camera.md)).
The MCP apply path already does this for every editing command
(`apply_as_agent` in `mcp/mod.rs`), and the panel's entries do the same.

### The proposal

While a proposal is open, the node has a **proposed camera**: a fitted
`CameraIntrinsics` and its report, held as viewer state the way the Move Camera
lock holds a pending pose. It is not in the version history until it is
applied. It carries a **generation**, a counter that every change of its fit
inputs advances; a change that leaves the inputs as they were keeps the
generation.

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
  which sets the domain to its angle. The default stays the corner. The
  keypoints are read once, when the proposal opens, on the GUI thread, as the
  Refit Spline dialog reads them. Their angle is re-read from the keypoint's
  pixel under each new fit, which reads no file.
- **Preview retriangulation**, below.
- **Apply** and **Cancel**.

Every change refits at once, on the GUI thread: the lens fit and the panel's
`Derived::compute` for the proposed column take milliseconds, as they do for
the Refit Spline dialog. The fit report is one line under the strip, for
example: "rms 0.13 px, radial 0.013 px, max 0.29 px over θ ≤ 84.5°; fx/fy
aspect 0.9978 dropped". A spline fit is constrained to stay monotone, so it has
an inverse; where that constraint bound, the line adds its range, for example
"monotone constraint bound at 1 angle, 113.2°", because there the proposed
curve is the closest invertible one rather than the current one. The
observation comparison and the preview's triangulation need the reconstruction,
and are computed by the proposal's job (§ "The job" below).

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
  going. The range where the monotonicity constraint bound is marked on the θ
  axis, so a departure there reads as the constraint rather than as a poor
  fit.
- **An observation rug** under the shared θ axis: one tick per observation of
  the camera's images, at its incidence angle under the current model. It shows
  where the data is: the Kerry rug thins out at 86°. Beside the rug is the count
  of observations past the current model's trusted bound.

### Ending a proposal

**Apply** runs the shipped direct switch
([`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md))
with the proposal's request, on the GUI thread. It pushes one version with the
label that edit writes, `Switched camera 0 of kerry_park from OPENCV_FISHEYE to
SFMTOOL_FISHEYE` for a change of model and `Refit spline of camera 0 of
kerry_park: 8 → 12 coefficients, domain 150.1° → 108.8°` for a refit, and the
Action Log row that edit writes. **Cancel** and `Escape` discard the proposal.

The implicit endings follow the Move Camera lock: selecting another node,
closing it, or an MCP edit tool naming it applies the proposal when the fit
succeeded, since an edit has a history and undo can reverse it. The proposal is
dropped when the fit was refused. Closing is the one ending that cannot apply
afterwards, so it applies before the close is asked about, as the lock commits
before it.

Three other states end or block a proposal:

- **Undo, redo or a history jump on the node**, from the menus or the wire,
  drops the proposal, as Cancel does, and writes a `Kind::View` row saying so.
  The proposal was fitted to the camera of the version it opened on, and once
  the cursor moves it describes a change to a camera the node no longer shows.
  Applying it first instead would put a version on the history that the undo
  then steps back out of, which is not what the person asked for.
- **Other edits on the node from the menus** are greyed while a proposal is
  open, with "Apply or cancel the camera model proposal first" as the hover
  text. That includes entering the Move Camera lock on the node and every
  operation that would start a background task on it, so no background task
  can start on a node with a proposal open. An MCP edit tool naming the node
  applies the proposal first, by the implicit rule, and then runs.
- **Hiding the node** leaves the proposal and its preview as they are. A hidden
  node keeps its GPU buffers (`scene_renderer/mod.rs`), so showing it again
  shows the preview where it stands.

Every ending of the proposal ends its preview in the same frame: the preview
buffers are dropped and the node's points are drawn at their stored positions.
Unticking the checkbox is the way to see the motion back. Apply in particular
does not wait for it: the switch moves no point, so the version Apply pushes
holds the stored positions, and the new base it installs recreates the node's
point buffer from them (`upload/points.rs`, through the base-identity check in
`app.rs`). A preview that was ticked therefore returns to the stored positions
in one frame when Apply is pressed, which is the truthful picture of what Apply
did. The MCP reply of `switch_camera_model { proposal: true }` is an edit's
reply and comes at once.

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
  5° from θ_fit to the edge). For Kerry that shows the band from 86° to the
  image circle and how it is now divided.
- **An observation layer.** For each observation of the image, a tick at the
  keypoint and a line to its reprojection under the current model and under the
  proposed one. This is the fixed-set comparison, shown where it happens.

The axes and rings follow the proposed model while the proposal is open, and a
checkbox in the ⚙ popup puts them back on the current model.

**The proposal banner.** While a proposal is open, the 3D view draws a banner
top-centre in the style of the Move Camera lock banner
([`../gui/viewport-hud.md`](../gui/viewport-hud.md) § "The lock banner borrows
the style"), and like it outside the HUD panel. Its first line names the
proposal: the camera, the node, the source and target models and the
coefficient count. While Preview retriangulation is ticked, a second line
carries the preview's counts: the proposal's tracks, how many moved by the
difference rule, how many gained a position, how many lost one, and the median
and 90th-percentile displacement of the moving finite points, in the
reconstruction's units. The count that lost a position is broken down by the
proposed run's outcome, largest first: `Kept`, or the `PointVerdict` of a
`Solved` direction. For example: "37 lose their position: 31 behind a camera,
6 too few usable rays". The line is drawn in the failure colour when the count
is not zero. The reasons are worded as the Retriangulate edits' Action Log
words them, from `PointVerdict::label` and the sentence for a kept point, so
the banner, the point info and the wire say the same thing about one point.
While the job has no answer for the current generation the second line reads
"Triangulating…".

### The 3D view

The switch changes how rays map to keypoints and leaves every point where it
is, so the 3D effect of a new model is not something Apply shows. It is what
retriangulating the same keypoints through the new model would give: where the
tracks would be if they were solved again, and so where a bundle adjustment
after the switch would start from. The proposal strip carries a **Preview
retriangulation** checkbox, off when a proposal opens. Ticking it moves the
points and surfels of the proposal's tracks, in the 3D view, smoothly from
where they are to where the proposed model puts them. Unticking it moves them
back the same way. The motion is the point of the preview: which parts of the
cloud move, in which direction and how far is visible in the second the move
takes, in a way two static clouds side by side do not show.

The checkbox is greyed, with `retriangulate_points`' refusal sentence as its
hover text, where the node cannot be retriangulated: a `sift_files` node
without the inline keypoint column, whose observations carry no pixel.

**The proposal's tracks.** Every track with at least one observation in an
image taken through the proposal's camera. A track only other cameras observe
reads the same rays under both models and does not move.

**Two triangulations.** The job triangulates the node's points twice, with
`retriangulate_points` over `All` and the default options the viewer's
Retriangulate edits use: once through the current cameras and once through the
current cameras with the proposed one in place. Both runs start from one
materialisation of the same version, so their statuses and answers index the
same rows, and the materialisation's own row map takes each row back to the
point the 3D view draws: a row of the base's point buffer for a base point and
a row of the additions buffer for an added one.

**What each point does.** The outcomes of the two runs decide it, looked up by
index with `RetriangulateReport::point`. An outcome is a finite position when
it is `Solved` with `PointVerdict::Finite`, `FinitePruned`, or `Ranged` at a
finite distance; it is a direction when it is `Solved` with `Marked`, `Thin`,
`NoDepth`, `Behind`, `OverBar`, or `Ranged` at an infinite distance. Under the
default options the floor and the likelihood rule are off, so no point is
called `Thin` or `NoDepth`, and a point stored as a direction carries its own
`w` as the incoming mark and is answered as a direction under both models.

- **Both finite: the difference rule.** The point moves from its stored
  position by the difference between the two answers:

      target = stored + (proposed answer − current answer)

  The motion starts at the stored position, so ticking the box causes no jump.
  The stored points usually come out of a bundle adjustment, so a fresh
  triangulation moves them a little under either model; taking the difference
  of two triangulations of the same tracks keeps that movement out, and leaves
  the model as the only cause of the motion.
- **Gains a position**: the proposed status is a finite position and the
  current one is not, because the current model's rays put the point behind a
  camera observing it, or because fewer than two of its observations gave a
  usable ray. There is no current answer to take a difference from, so the
  point moves from its stored position straight to the proposed answer.
- **Loses its position**: the current status is a finite position and the
  proposed one is a direction or `Kept`. There is no proposed answer to move
  toward, so the point stays where it is and its colour moves to the
  **failure colour** as the motion runs (§ "The motion"). These are the points
  the reviewer most needs to find: a track the current model triangulates and
  the proposed one cannot is the plainest sign that the proposed model is wrong
  somewhere, so the preview makes them stand out rather than recede. The point
  info of a selected point in the failure colour gives the reason from the
  proposed run's outcome.
- **Neither finite** (a stored finite point that both runs answer as a
  direction, or keep): there is no position under either model to take a
  difference between, and the point does not move.
- **Held**: never read by either run, and does not move.

The formula is only applied where both answers are positions, so an absent or
non-finite answer under either model never reaches it: a point absent under
the proposed model is the loss case, absent under the current model the gain
case, and absent under both stays.

On Kerry the gain case is narrower than it may look. The `OPENCV_FISHEYE`
inverse returns a finite ray past its trusted range (the blend toward the
identity in `camera/distortion/kernels/blend.rs`, and the identity ray itself
where the Newton recovery does not converge), so a peripheral observation has
a ray under the current model, a wrong one. A peripheral track therefore has a
current answer, usually a finite one, and follows the difference rule: its
motion is the correction of those wrong rays. Since both models give every
keypoint a ray, the number of usable rays is the same under both, and a Kerry
track gains a position only where the current model's rays put it behind a
camera and the proposed model's do not.

**Directions and ranged points rotate.** A difference of positions is not the
right motion for a point whose distance is not free:

- A **point at infinity** has a bearing. Its target is the stored bearing turned
  by the rotation that takes the current answer's direction to the proposed
  answer's, along the shorter arc.
- A **ranged point** keeps its distance from its reference image's camera
  centre. Its target is the stored position turned about that centre by the
  rotation that takes the current answer's direction from the centre to the
  proposed answer's, so it keeps its distance. The shader blends along the
  straight line between the two, so in the middle of the motion the point sits
  slightly inside that sphere; both ends are on it.

**Surfels.** Each surfel moves with its point and keeps its orientation. Its
size changes by the patch-frame factor for a move from the stored position to
the target, the ratio of placement distances
([`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
§ "The frame follows the depth"), so it keeps its angular size as it moves. A
rotated bearing keeps its size. A surfel whose point loses its position takes
the failure colour with it.

### The job

No worker computes intrinsics or comparisons in the viewer today, and this one
is not a background task. Background tasks run one at a time across the viewer
and lock their node
([`../gui/background-tasks.md`](../gui/background-tasks.md)), and a proposal is
neither: it pushes nothing, so it has nothing to lock, and a
proposal job must not hold off a bundle adjustment on another node or be held
off by one. It follows the bench's live evaluation
([`../gui/bench.md`](../gui/bench.md) § "Live evaluation",
[`bench/live.rs`](../../crates/sfm-explorer/src/bench/live.rs)) instead:

- **Its inputs** are the proposal's generation and the version's
  `VersionSerial`. Nobody asks for a job: once per frame the viewer finds a
  proposal whose inputs have no answer, and starts one.
- **One job runs at a time.** When the inputs change while one runs, it is
  cancelled through its `Progress`, and the next starts once the cancelled one
  has reported back. An answer is installed only when its inputs are still the
  proposal's, so an answer for an older generation is discarded rather than
  moved to.
- **It locks nothing, pushes no version and writes no Action Log row**, and the
  Background Task panel does not show it.

One job does, in order:

1. **The switch.** `switch_camera_model` over the materialisation, for the
   proposal's camera and request, with the outermost keypoint left out of the
   report, so no `.sift` file is read. Its report carries the observation
   comparison, which the job sends to the panel as soon as the switch returns,
   and its value is the switched value the second run reads. It takes no
   `Progress`, so a cancel is read when it returns; it is a fit and a pass over
   the camera's observations, which is milliseconds.
2. **The current-model run**, only when the proposal has none. It does not
   depend on the controls, so it is computed once per proposal, by its first
   job that reaches this step, and kept for the proposal's life. Every way of
   changing the version ends the proposal, so it cannot go stale.
3. **The proposed run** over the switched value.

Steps 2 and 3 run only while Preview retriangulation is ticked, or an MCP call
has asked for the preview. An answer is kept by generation, so unticking and
ticking again without a change of control reuses it rather than solving again.

**Cost.** The materialisation is made once per proposal, and is the base
itself, shared by `Arc`, when the version's overlay is empty. Each job then
makes two whole-value copies: the switched value `switch_camera_model` builds
(`clone_for_edit`), and the answer value `retriangulate_points` over `All`
builds. The current-model run's copy is made once. The preview reads only the
statuses and positions of the proposal's tracks from the answers, and drops the
rest. `These` would name only those tracks, but it rebuilds the addition set's
derived indexes once per point and is meant for a handful, so the job solves
every point. Whether two copies per job is too slow on a large node is an open
question below. The write-back's placement distance is the other cost:
`ImageTable::placement_scale` sums the camera centres on every call
(`image_table.rs`), which makes a whole-value write-back cost points × images.
The core addition in Amends computes that centroid once per call, and the
preview's size factors use the same form.

### The motion

A blend value `s` runs from 0 (stored) to 1 (target), and each moving point is
drawn at `stored + s · (target − stored)`, normalised for a bearing. A point
that loses its position is drawn at its stored position, at full opacity, in
`mix(colour, failure colour, s)`, so the red builds up while the other points
travel and recedes when the box is cleared. The failure colour is a saturated
red, `[235, 30, 30]`, held in one constant beside `TINT_PALETTE`
(`crates/sfm-explorer/src/scene.rs`). It is not in that palette, whose nearest
entry is Vermillion `[213, 94, 0]`, and it replaces the node's tint for these
points, so a node tinted Vermillion still shows its failures.

`s` is driven one frame at a time by the controller Maintain Z-up uses for its
turn ([`../gui/viewport-navigation.md`](../gui/viewport-navigation.md) §
"Maintain Z-up"), generalised. `righting::step` today carries an unsigned speed
(`speed.max(0.0)`), turns toward a goal fixed at +Z, reads an unsigned angle,
uses its module constants `MAX_SPEED` (4) and `ACCELERATION` (20), and is
`pub(super)` to `viewer_3d`. Its one-dimensional core moves out into a
`pub(crate)` step function that both callers reach:

    step(value, velocity, goal, dt, max_speed, acceleration)
        -> (value, velocity)

The velocity carries a sign. The step accelerates toward the goal at
`acceleration`, caps the speed at `max_speed`, slows so that the value stops
exactly at the goal, and when the velocity points away from the goal it first
slows through zero at `acceleration` before turning. Maintain Z-up calls it
with the angle to +Z as the value, a goal of 0 and its own two constants, and
turns `world_up` by the change, so its timings are unchanged. Its pause rule
(a paused turn restarts from rest) stays its own. The preview calls it with `s`,
a goal of 0 or 1, a cap of 2 per second and an acceleration of 10 per second²,
so a full move takes 0.7 s, with 0.2 s ramps at each end. Unticking while the
motion runs sets the goal to 0, and the signed velocity slows the points and
sends them back without a jump.

**A new answer while the box is ticked** comes after a control changed, and
gives the moving points new targets. Each point's drawn position is
`stored + s · (target − stored)`, so swapping the targets at any `s` above 0
would make the points jump. The new answer is therefore held as pending, and
the goal is set to 0: the points run back to their stored positions on the old
targets, the preview buffer is rewritten with the pending targets when `s`
reaches 0, and the goal is set back to 1. A newer answer arriving during the
run back replaces the pending one. When `s` is already 0 the targets are
written at once.

### What the preview does not change

The value: every panel other than the 3D view reads the stored points, the
point info, Track View and Image Detail's reprojections among them. The CPU
side of the 3D view does too: Go to Point frames a point at its stored
position, and the bench's figures read stored positions. The one CPU-drawn
part of the 3D view that would visibly disagree with a moving point is its
track rays (`upload/track_rays.rs`), which end at the point. While `s` is above
0, the track rays of a selected point that the preview moves are not
drawn, and they return when `s` is back at 0.

The GPU side follows the drawn position. The pick pass runs the same vertex
shader, so a click on a moving point picks it where it is drawn and selects the
point it is, and Alt+click reads the orbit target's depth from the drawn
position. The camera frustums and the background mesh stay on the current
model, because they would change by fractions of a pixel, which are not visible
at viewport scale.

### Rendering

`PointInstance` is 16 bytes and its buffer is `VERTEX` only (`gpu_types.rs`,
`upload/points.rs`), so the target does not go into it. A node with the preview
on has a separate **preview buffer**: one `vec4<f32>` per row of its point
buffer, `VERTEX | COPY_DST`, stepping per instance beside `PointInstance` and
the deleted mask's `point_alive_buffer`, which is the same kind of buffer. Its
`xyz` is the target (a unit direction for a bearing) and its `w` a flag: 0 for
a row that does not move, 1 for a moving row, and 2 for a row that stays where
it stands and takes the failure colour. A row that does not move is marked by
its flag, not by a zeroed position. The overlay's additions buffer
(`upload/additions.rs`) gets a preview buffer of its own, by the same rule. A preview buffer is created when the
preview is first ticked on the node and dropped when the preview ends; while
one exists the node is drawn with a variant of the point pipeline that reads
it, and every other node with the pipeline it has today.

`s` goes in `ReconUniforms`, in the `_pad` that follows `show_infinity`, so the
struct's size does not change. The struct is declared again in each shader that
reads it (`points.wgsl`, `patch.wgsl`, `frustum.wgsl`, `image_quad.wgsl`,
`distorted_quad.wgsl`), and every copy names the new field. The additions
bundle writes its own `ReconUniforms`, so `s` is written there as well.

Surfels ([`../gui/patch-rendering.md`](../gui/patch-rendering.md)) are packed by
slot rather than by point, so their preview buffer is written by slot, through
`slot_of_point`, the way the deleted mask's `alive_buffer` is. Each slot holds
its target centre and the flag in a `vec4<f32>`, and the size factor that
scales `u_halfvec` and `v_halfvec` by `mix(1, factor, s)`. The additions'
surfels get the same through their own slot map.

A new answer is one upload of each preview buffer. A frame of motion uploads
nothing but `s`. The base's point and surfel buffers are never written, so
ending the preview restores nothing; it drops the preview buffers and sets `s`
to 0.

### Caches

Two caches key on `CameraRef` alone and must not serve the proposed camera as
the current one:

- the panel's `derived` map (`intrinsics_detail/mod.rs`);
- the Image Detail layer cache (`image_detail/mod.rs`).

Both gain a second key for the model they were computed from: current, or the
proposal's generation. Apply goes through `forget_recon` the way every bulk
edit does.

## MCP

The switch differs from Move Camera in one way that matters for the wire: its
inputs are a few discrete choices an agent can state. So the proposal is on the
wire too, and an agent can open it, screenshot the panel and the overlay, and
then apply or cancel. This is the check a reviewer would make. Every argument
takes the name `switch_camera_model` already gives it, and none is a bare
`model` ([`../GLOSSARY.md`](../GLOSSARY.md) § "Camera intrinsics").

- **`propose_camera_model { reconstruction_label, camera_intrinsics_index,
  camera_model?, coeff_count?, theta_fit_deg?, spline_domain_deg?,
  preview_retriangulation?, animate? }`** opens or replaces the proposal
  exactly as the panel does, with the defaults `switch_camera_model` gives each
  argument, and answers with the fit report and the `proposed` block below. It
  pushes nothing. A call whose fit inputs equal the open proposal's keeps the
  proposal and its generation, so a call that only changes
  `preview_retriangulation` or `animate` neither refits nor solves again. It is
  refused on a busy node, with the busy sentence, and it ends a held Move
  Camera lock on the node first, as an edit tool does.
- **`cancel_camera_model_proposal { reconstruction_label }`** is Cancel.
- **`switch_camera_model`** exists and applies a direct switch or a spline
  refit ([`../gui/mcp-server.md`](../gui/mcp-server.md)). It gains a
  `proposal: true` form that applies the open proposal instead of fitting one.
  The direct form is an edit tool naming the node, so it applies an open
  proposal first by the implicit rule; an agent that wants the direct switch
  alone cancels first. Either answers like every edit tool: the version, its
  label, the sentence recorded, and the `fit` object, including the
  observation comparison.

`get_camera_intrinsics` gains a `proposed` block while a proposal is open: the
proposed camera, its fit report, the `generation`, the observation comparison
once the job has sent it, `preview_retriangulation`, the current `s`, and a
`triangulation` block. `get/set_image_detail_display` gains the field mode.

The `triangulation` block's `state` is `off` while the preview is not asked
for, `pending` while the job has no answer for the current generation, `ready`
with the counts the banner shows (the proposal's tracks, moved by the
difference rule, gained a position, lost one, and the median and 90th
percentile of the displacement, in the reconstruction's units, plus
`lost_points`: each point that lost its position, by index, with its reason as
`kept` or the verdict's wire code and label, so an agent can select one and
screenshot it), or `refused`
with `retriangulate_points`' sentence.

`propose_camera_model` replies before the job has run, so its `triangulation`
block is usually `pending`. An agent waits by polling `get_camera_intrinsics`
until the state is `ready` and the block's `generation` is the one the call
returned. The 200 ms-or-handle reply of `add_camera_image_to_tracks` is not
used, because that handle names a background task and this job is not one.

`animate` defaults to true. With `animate: false`, `s` is set to its goal at
once instead of being stepped: to 1 when the preview is ticked and to 0 when it
is cleared. Before the job has an answer there is nothing to set it toward, so
`s` jumps to 1 in the frame the answer lands, and a new answer under
`animate: false` replaces the targets at once rather than running back first.
An agent's next screenshot after the state reads `ready` shows the end state
without waiting out the motion.

## Testing

- **Viewer lib tests:** a proposal leaves the history unchanged; Apply pushes
  one version with the shipped edit's label; Cancel restores; the two caches do
  not cross; undo, redo and a history jump on the node drop an open proposal
  and write one `Kind::View` row; the node's other menu edits and Move Camera
  are greyed while a proposal is open; Switch Model… is greyed on a busy node;
  opening a proposal ends a held Move Camera lock by its implicit rule; hiding
  the node leaves the preview's state unchanged.
- **Preview computation tests**, on a synthetic two-camera scene:
  - a proposal whose camera is the current camera itself, given to the job
    directly rather than fitted, moves no point and counts no gain or loss. A
    fitted proposal cannot test this, since a refit is not bit-equal to its
    source;
  - a proposal for one camera moves only the tracks that camera's images
    observe;
  - a node with an overlay holding additions and deletions: both runs read one
    materialisation, a moving added point's target lands in the additions
    buffer's row, and a deleted point gets no row;
  - a held point does not move; a ranged point keeps its distance from its
    reference camera centre at the target; a point at infinity rotates and
    keeps unit length;
  - a point the proposed run keeps, and one it answers as a direction, are
    counted as losing a position and are drawn where they stand, in
    `mix(colour, failure colour, s)`, over a node tinted Vermillion as well;
    `lost_points` lists them with their reasons; a
    point the current run answers behind a camera and the proposed run finite
    moves to the proposed answer;
  - the current-model run is computed once per proposal across several
    generations, and no `.sift` file is read by a job.
- **Preview motion tests:** the generalised step reaches the goal and stops
  there without overshoot, reverses without a jump in `s` when the goal flips
  mid-motion, and reproduces Maintain Z-up's existing timings when called with
  its constants; the stored positions are never written; an answer from an
  older generation is not moved to; a new answer arriving at `s = 1` runs `s`
  down to 0 without a jump, rewrites the preview buffer only at 0, and runs
  `s` back up to 1 on the new targets; every ending of the proposal drops the
  preview in the same frame.
- **Upload tests** in `scene_renderer/upload/tests.rs`, on the `noop` backend:
  the preview buffers are created at the point buffer's and the additions
  buffer's row counts with `COPY_DST`, a non-moving row carries flag 0, the
  surfel preview buffer is written by slot through `slot_of_point`, `s`
  reaches both bundles' `ReconUniforms`, and the buffers are dropped when the
  preview ends.
- **MCP tests:** `propose_camera_model` pushes nothing and answers with the fit
  report and a `pending` triangulation block; a second call with the same fit
  inputs keeps the generation; `get_camera_intrinsics` reports `ready` once the
  job lands; `animate: false` given before the answer leaves `s` at 0 until the
  answer lands and then sets it to 1 in one frame; `switch_camera_model {
  proposal: true }` applies the open proposal; a direct switch applies an open
  proposal first; `cancel_camera_model_proposal` drops one; no tool takes an
  argument named `model`.
- **A Kerry evaluation run**, kept as a script rather than a test:
  1. switch tk107 → Add Image to Tracks over all images;
  2. bundle adjust with the spline released;
  3. report the observation counts by θ band before and after, and the fitted
     spline against the image circle at 245 px.

  The result is judged in the viewer.

## Order of work

1. The proposal in the Camera Intrinsics panel: the controls, the two-model
   plots, the change plot and the rug. Apply as the shipped edit. The endings,
   including undo and the greyed entries.
2. The Image Detail change field and observation layer.
3. Preview retriangulation. It needs the two core additions in Amends: the
   outermost-keypoint option on `switch_camera_model` and the placement
   distance from a centroid computed once. Then the job, the generalised step,
   the preview buffers and the banner.
4. `propose_camera_model`, `cancel_camera_model_proposal`, the proposal form of
   `switch_camera_model` and the `proposed` block.

## Open questions

- **The dropped aspect.** `SFMTOOL_FISHEYE` has one focal. The Kerry fx/fy
  differences of 0.2 % and 0.7 % are most likely fitting noise on square
  pixels, and the switch drops them visibly. If a real lens needs them, an
  aspect parameter is a format change to the spline models. Measure after the
  Kerry run whether bundle adjustment's residuals show an azimuthal pattern.
- **Whether Apply should be able to retriangulate.** The preview shows where
  the tracks would be triangulated, and Apply leaves them where they are.
  The recommended sequence after the switch starts with a bundle adjustment,
  which re-reads the points itself, so the switch stays a lens-only edit. If
  switching and then triangulating again turns out to be the usual sequence,
  an "Apply and retriangulate" choice that pushes the switch and the
  triangulation as one version could be added.
- **Whether the preview needs an exaggeration.** Where the model change moves
  points by much less than their spacing, the motion may be too small to see.
  A factor `k` in `target = stored + k · (proposed − current)`, shown in the
  strip whenever it is not 1, would make it visible; whether it is needed
  should be found on Kerry first.
- **Very large displacements.** A track whose observations are off by hundreds
  of pixels, as some of Kerry's peripheral ones are, can move across the scene
  under the difference rule, and a few of those can dominate what the eye sees
  in the motion. Whether such points should be capped, faded or listed apart
  in the banner should be decided after seeing them on Kerry.
- **A colour for points that gain a position.** They could ease toward a
  second colour, such as green, while they move, so both kinds of change show
  by colour as well as by motion. Most moving points already show their change
  by moving, so this waits until the red has been seen on Kerry.
- **Whether two whole-value copies per job is fast enough.** Each job copies
  the value twice (§ "The job"). If that is too slow on a large node, the core
  addition that removes both is a way for `retriangulate_points` to solve
  through a camera table the caller states and return its statuses and
  positions without building a value. That would be a mode of the existing
  function, with the same rules, not a second triangulation.
- **Whether other previews share the motion.** The Refit Spline dialog changes
  a camera too, and
  [`move-camera-preview-amendment.md`](move-camera-preview-amendment.md)
  previews a re-triangulation under Move Camera by writing over the base's
  rows. Both could use the same preview buffers and blend.
- **Whether the switch should release the principal point** for a model whose
  source never fitted it. Today nothing releases it anywhere, and the switch
  keeps that.

The regularization weight is an open question of the fit itself, in
[`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md).
The spline domain is settled there: the default is the far image corner, the
model's own reach, and the outermost keypoint with its button is shown wherever
the domain is edited, so a circular fisheye is trimmed to its image circle by
choice.
