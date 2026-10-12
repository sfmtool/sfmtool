# Previewing a camera model proposal

**Status:** Draft

The viewer's camera model proposal is built: **Switch Model…** in the Camera
Intrinsics panel opens it, its strip of controls refits the camera on the GUI
thread, the Parameters and Derived tables show the current and the proposed
camera side by side, and **Apply** pushes the switch and the retriangulation of
the camera's tracks as one version, through the `Switch camera model`
background task
([`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md),
[`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md) § "The proposal
strip"). The comparison is built too: the projection plot draws both cameras
with a plot of the change and an observation rug
([`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md) § "4. Projection
plot"), and the Image Detail intrinsics layer's **Change to proposed** field
draws the change on the photograph with its observation layer
([`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md) § "The Change to
proposed field"). On the wire, `propose_camera_model`,
`cancel_camera_model_proposal`, the `proposal: true` form of
`switch_camera_model`, the `proposed` block of `get_camera_intrinsics` and the
`field` of `get/set_image_detail_display` are built
([`../gui/mcp-server.md`](../gui/mcp-server.md) § "`propose_camera_model` /
`cancel_camera_model_proposal`").

This draft is what remains, **the preview**: a job that runs the switch and the
retriangulation of the proposal's tracks off the GUI thread while the proposal
is open, its observation comparison sent to the panel, a **Preview
retriangulation** checkbox that moves the points and surfels of those tracks
smoothly in the 3D view to where Apply will put them, a banner with the
preview's counts, and Apply installing the job's answer instead of solving
again.

Decided in outline:

- While a proposal is open, a **Preview retriangulation** checkbox moves the
  points and surfels of the proposal's tracks smoothly, in the 3D view, to
  where Apply will put them, and back when it is cleared.
- The job is not a background task: it locks nothing, pushes nothing and is not
  shown in the Background Task panel (§ "The job").

Not decided: see [Open questions](#open-questions).

Builds on the core pieces made for it, which are built and specified:
`retriangulate_points`
([`../core/reconstruction/triangulation-rules.md`](../core/reconstruction/triangulation-rules.md)
§ "The report"), whose report carries a `RetriangulatedPoint` for every point
the call was asked about, with its `index`, its `new_index` and a
`RetriangulateOutcome` (`Held`, `Kept`, or `Solved` with the `PointVerdict`
that decided it); `RetriangulateWhich::TheseInBulk`, the bulk path Apply
already uses; `SwitchCameraModelOptions::outermost_keypoint`
([`../core/reconstruction/switch-camera-model.md`](../core/reconstruction/switch-camera-model.md)
§ "The outermost keypoint"); and `ImageTable::camera_centroid`,
`ImageTable::placement_scale_from` and the public `patch_frame_factor`
([`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
§ "The frame follows the depth"). This draft reads losses of a position off
the retriangulation's outcomes and adds no verdict logic of its own.

Amends:

- [`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md) § "The
  proposal strip", which gains the observation comparison the job sends.
- [`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md),
  whose Apply gains the installation of the job's answer, and whose endings end
  the preview in the same frame.
- [`../gui/mcp-server.md`](../gui/mcp-server.md), whose `propose_camera_model`
  gains `preview_retriangulation` and `animate`, and whose `proposed` block
  gains the observation comparison, `preview_retriangulation`, `s` and the
  `triangulation` block.
- [`../core/reconstruction/switch-camera-model.md`](../core/reconstruction/switch-camera-model.md)
  § "The outermost keypoint", which names the proposal's job as a caller that
  leaves the keypoint out.
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
| cam0 | 129.523 | 0.013 px | 0.130 px | 0.292 px | 84.5° | aspect fy/fx 0.9978 |
| cam1 | 129.301 | 0.003 px | 0.384 px | 0.690 px | 83.8° | aspect fy/fx 1.0067 |

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

### The observation comparison

The panel's comparison of the two cameras is built
([`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md) § "4. Projection
plot", "The Change to proposed field"). What it does not yet show is the
observation comparison: the median, 90th percentile and maximum error of the
camera's observations under each model over one fixed set, and the count past
the trusted bound. It needs the reconstruction's pixels, and is computed by the
proposal's job (§ "The job"), which sends it to the panel when the switch
returns.

### The proposal banner

While a proposal is open, the 3D view draws a banner
top-centre in the style of the Move Camera lock banner
([`../gui/viewport-hud.md`](../gui/viewport-hud.md) § "The lock banner borrows
the style"), and like it outside the HUD panel. Its first line names the
proposal: the camera, the node, the source and target models and the
coefficient count, and for a same-model target the count and domain end before
and after, as the refit's version label names them. While Preview
retriangulation is ticked, a second line carries the preview's counts: the
proposal's tracks, how many move, how many lose their position, how many are
left where they were for too few usable rays, and the median and
90th-percentile displacement of the finite points that move, in the
reconstruction's units. The count that lose their position is broken down by
the proposed run's `PointVerdict`, largest first. For example: "37 lose their
position: 37 behind a camera that sees them". The line is drawn in the failure
colour when that count is not zero. The reasons are worded as the Retriangulate
edits' Action Log words them, from `PointVerdict::label` and the sentence for a
kept point, so the banner, the point info, the wire and the Action Log row
Apply writes say the same thing about one point. While the job has no answer
for the current generation the second line reads "Triangulating…".

### The 3D view

The switch changes how rays map to keypoints, and Apply then retriangulates
the proposal's tracks through the new model, so the 3D effect of a proposal is
where those tracks are solved again, which is also where a bundle adjustment
after Apply starts from. The proposal strip carries a **Preview
retriangulation** checkbox, off when a proposal opens. Ticking it moves the
points and surfels of the proposal's tracks, in the 3D view, smoothly from
where they are to where Apply will put them. Unticking it moves them back the
same way. The motion is the point of the preview: which parts of the cloud
move, in which direction and how far is visible in the second the move takes,
in a way two static clouds side by side do not show.

The checkbox is greyed, with `retriangulate_points`' refusal sentence as its
hover text, where the node cannot be retriangulated: a `sift_files` node
without the inline keypoint column, whose observations carry no pixel. On such
a node the job still runs its first step, the switch, which reads the
observations' pixels from the `.sift` files (§ "The job"), and Apply pushes the
switch alone ([`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md)
§ "Ending a proposal").

**The proposal's tracks.** Every track with at least one observation in an
image taken through the proposal's camera. These are the points Apply
retriangulates. A track only other cameras observe reads the same rays under
both models, is not retriangulated, and does not move.

**One triangulation.** The job runs `retriangulate_points` over
`TheseInBulk(the proposal's tracks)`, with the default options the viewer's
Retriangulate edits use, on the switched value: the materialisation of the
version with the proposed camera in place. That is the run Apply pushes, so its
answer is the version Apply would install. The switch renumbers nothing, so the
report's `index` is a row of the materialisation, and the materialisation's own
row map takes each row back to the point the 3D view draws: a row of the base's
point buffer for a base point and a row of the additions buffer for an added
one. No run through the current camera is needed: every case below compares
the proposed answer with what the value stores, which is what Apply replaces.

**What each point does.** The proposed run's outcome decides it, looked up by
index with `RetriangulateReport::point`, against the point's stored geometry.
An outcome is a finite position when it is `Solved` with `PointVerdict::Finite`,
`FinitePruned`, or `Ranged` at a finite distance; it is a direction when it is
`Solved` with `Marked`, `Thin`, `NoDepth`, `Behind`, `OverBar`, or `Ranged` at
an infinite distance. Under the default options the floor, the likelihood rule
and the reprojection bar are off, so a stored finite point becomes a direction
only by `Behind`, and a point stored as a direction carries its own `w` as the
incoming mark and is answered as a direction. For a point that moves the target
is what Apply writes for it, and the motion starts at the stored position, so
ticking the box causes no jump.

- **Moves**: the outcome is a finite position, or a direction for a point
  stored as one. The target is the proposed answer. A finite point is drawn on
  the straight line from its stored position to the answer. A point at
  infinity is drawn on the same line between its stored bearing and the
  proposed one, normalised, so it turns along the shorter arc. A ranged point
  keeps its distance from its reference image's camera centre at both ends;
  between them the straight line puts it slightly inside that sphere.
- **Loses its position**: the point is stored finite and the outcome is a
  direction. Apply writes it as that direction, a point at infinity. A point
  cannot travel to infinity along a line, so the preview leaves it where it
  stands and moves its colour to the **failure colour** as the motion runs
  (§ "The motion"); when the version lands it is drawn where the 3D view draws
  any point at infinity. These are the points the reviewer most needs to find:
  a track the proposed model's rays put behind a camera that sees it is the
  plainest sign that the proposed model is wrong somewhere, so the preview
  makes them stand out rather than recede. The point info of a selected point
  in the failure colour gives the reason from the proposed run's outcome.
- **Left where it was**: the outcome is `Kept`, because fewer than two of the
  point's observations give a usable ray. Apply keeps its stored geometry, and
  the preview does not move it or colour it. A track with fewer than two
  observations is kept under any model, so this outcome says nothing about the
  proposed model by itself; the banner counts it apart.
- **Held**: never read by the run, unchanged by Apply, and does not move.

The answers usually differ a little from the stored points even where the
model hardly changes, because the stored points come out of a bundle
adjustment and a fresh triangulation does not reproduce it. That motion is
part of what Apply does, and the preview shows it with the rest.

On Kerry the `OPENCV_FISHEYE` inverse returns a finite ray past its trusted
range (the blend toward the identity in `camera/distortion/kernels/blend.rs`,
and the identity ray itself where the Newton recovery does not converge), so a
peripheral observation has a ray under the current model, a wrong one, and a
peripheral track was placed through it. Its motion is the correction of those
rays, and it can be large where they were badly wrong. A track that moves far
is drawn like every other point, with no cap, fade or separate list: a far move
is either such a correction or a sign that the proposed model is wrong, and in
both cases it is what Apply does. Since both models give every keypoint a ray,
the number of usable rays is the same under both, and the proposed model leaves
no more Kerry points `Kept` than the current one would.

**Surfels.** Each surfel moves with its point and keeps its orientation. Its
size changes by the patch-frame factor for a move from the stored position to
the target, the ratio of placement distances that `patch_frame_factor` gives
([`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
§ "The frame follows the depth"), measured from the switched value's camera
centroid taken once, which is the factor the retriangulation
rescales the patch frame by, so the surfel keeps its angular size as it moves
and ends at the size Apply writes. A point at infinity keeps its size. A
surfel whose point loses its position takes the failure colour with it.

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
   proposal's camera and request, with
   `SwitchCameraModelOptions::outermost_keypoint` off, so it reads no `.sift`
   file for the outermost keypoint. Its report carries the
   observation comparison, which the job sends to the panel as soon as the
   switch returns, and its value is the switched value the second step reads.
   It takes no `Progress`, so a cancel is read when it returns. On a value
   with the inline keypoint column it is a fit and a pass over the camera's
   observations, which is milliseconds. On a `sift_files` value without that
   column it also reads the observations' pixels from the `.sift` files, on
   the job's thread, each file once, for every image that observes a point the
   camera's images observe, because the comparison and the points' stored
   errors are measured against those pixels. This step runs whether or not the
   Preview retriangulation checkbox is ticked or greyed. Where a file cannot be
   read the switch refuses (`SwitchCameraModelError::Observations`), and the
   job sends that sentence to the panel in place of the comparison.
2. **The retriangulation** of the proposal's tracks over the switched value,
   through `RetriangulateWhich::TheseInBulk`, since the proposal's tracks are
   far more than the handful `These` is built for.

Step 2 runs only while Preview retriangulation is ticked, or an MCP call has
asked for the preview. An answer is kept by generation, so unticking and
ticking again without a change of control reuses it rather than solving again.
The answer for the current generation keeps its value as well as its report,
since it is the version Apply installs (§ "Apply with the job's answer").

**Cost.** The materialisation is made once per proposal, and is the base
itself, shared by `Arc`, when the version's overlay is empty. Each job then
makes two whole-value copies: the switched value `switch_camera_model` builds
(`clone_for_edit`), and the answer value the retriangulation builds. The
answer for the current generation is held while the proposal is open, so a
proposal with the preview on holds one extra copy of the value. Whether two
copies per job is too slow on a large node is an open question below. The
write-back's placement distances cost one pass over the images per call, not
one per point: the retriangulation takes the camera centroid once
(`ImageTable::camera_centroid`), and the preview's size factors are measured
from one centroid the same way.

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

The motion is not exaggerated. Each point that moves ends at the position Apply
writes for it, however small the move, so what the preview shows at its end is
the version Apply pushes, except that a point that loses its position stands
where it was, in the failure colour, rather than at infinity.

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

### Apply with the job's answer

Where the proposal's job already holds the proposed run's answer for the
current generation (§ "The job"), that answer is the version, and Apply
installs it on the GUI thread without solving again: the same version the
`Switch camera model` task would push, with the same label and Action Log
row. Otherwise Apply starts the task, as it does now. The implicit endings
apply by the same rule.

Every ending of the proposal ends its preview in the same frame: the preview
buffers are dropped and the node's points are drawn at the positions the
node's value holds. Unticking the checkbox is the way to see the motion back.
Apply does not wait for the motion: the version it pushes holds the
retriangulated positions, and the new base it installs recreates the node's
point buffer from them (`upload/points.rs`, through the base-identity check in
`app.rs`). A preview that had run to its end hands over to the new base with no
visible change for the points that move, since their targets are the positions
Apply writes (§ "The 3D view", "What each point does"). A point that loses its
position goes from its stored position, in the failure colour, to where the 3D
view draws a point at infinity. Without the preview, the camera's tracks move in
one frame when the version lands. Where Apply runs as a background task, the
preview ends when the task starts, and the points stand at their stored
positions until it lands. The MCP reply of
`switch_camera_model { proposal: true }` is then an edit's reply at once when
the job's answer is installed, and the reply of a background edit otherwise.

## MCP

The tools are built, and gain:

- **`propose_camera_model`** takes `preview_retriangulation?` and `animate?`. A
  call that only changes either keeps the proposal's generation, so it neither
  refits nor solves again.
- The **`proposed` block** gains the observation comparison once the job has
  sent it (or the switch's `Observations` refusal sentence in its place),
  `preview_retriangulation`, the current `s`, and a `triangulation` block.

The `triangulation` block's `state` is `off` while the preview is not asked
for, `pending` while the job has no answer for the current generation, `ready`
with the counts the banner shows (the proposal's tracks, how many move, how
many lose their position, how many are left where they were, and the median
and 90th percentile of the displacement, in the reconstruction's units, plus
`lost_points`: each point that loses its position, by index, with the
verdict's wire code and label, so an agent can select one and screenshot it),
or `refused` with `retriangulate_points`' sentence.

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

- **Viewer lib tests:** Apply with the job's answer in hand installs that
  answer without solving again, and pushes the same version the task would;
  hiding the node leaves the preview's state unchanged.
- **Preview computation tests**, on a synthetic two-camera scene:
  - every target is the position the version Apply pushes holds for that
    point, and the answer value the job keeps is that version, equal to
    `switch_camera_model` followed by `retriangulate_points` over the same
    tracks;
  - a node with an overlay holding additions and deletions: the run reads one
    materialisation, a moving added point's target lands in the additions
    buffer's row, and a deleted point gets no row;
  - a held point does not move; a ranged point keeps its distance from its
    reference camera centre at the target; a point at infinity turns to the
    proposed bearing and keeps unit length;
  - a stored finite point the run answers as a direction is counted as losing
    its position and drawn where it stands, in `mix(colour, failure colour,
    s)`, over a node tinted Vermillion as well, `lost_points` lists it with its
    reason, and Apply writes it as that direction; a point the run keeps is
    counted as left where it was, not drawn in the failure colour, and keeps
    its geometry under Apply;
  - a job on a node with the inline keypoint column reads no `.sift` file; on
    a `sift_files` node without it, the job's switch reads each observing
    image's `.sift` file once and the checkbox is greyed with
    `retriangulate_points`' sentence, and a missing `.sift` file refuses the
    job's switch with the `Observations` sentence.
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
- **MCP tests:** `propose_camera_model` answers with a `pending` triangulation
  block; `get_camera_intrinsics` reports `ready` once the job lands;
  `animate: false` given before the answer leaves `s` at 0 until the answer
  lands and then sets it to 1 in one frame; `switch_camera_model { proposal:
  true }` with the answer in hand replies with the version at once.
- **A Kerry evaluation run**, kept as a script rather than a test:
  1. switch tk107 → Add Image to Tracks over all images;
  2. bundle adjust with the spline released;
  3. report the observation counts by θ band before and after, and the fitted
     spline against the image circle at 245 px.

  The result is judged in the viewer.

## Open questions

- **The dropped aspect.** `SFMTOOL_FISHEYE` has one focal. The Kerry fx/fy
  differences of 0.2 % and 0.7 % are most likely fitting noise on square
  pixels, and the switch drops them visibly. If a real lens needs them, an
  aspect parameter is a format change to the spline models. Measure after the
  Kerry run whether bundle adjustment's residuals show an azimuthal pattern.
- **Whether two whole-value copies per job is fast enough.** Each job copies
  the value twice (§ "The job"). If that is too slow on a large node, one way
  out is for `retriangulate_points` to solve through a camera table the caller
  states and return its statuses and positions without building a value. That
  would be a mode of the existing function, with the same rules, not a second
  triangulation. Apply would then solve again to build the version, rather
  than install the job's answer, so the saving is in the jobs a proposal's
  changes start, not in Apply.
- **Whether other previews share the motion.** A refit is a same-model
  proposal and has this preview already.
  [`move-camera-preview-amendment.md`](move-camera-preview-amendment.md)
  previews a re-triangulation under Move Camera by writing over the base's
  rows, and could use the same preview buffers and blend.

The regularization weight is an open question of the fit itself, in
[`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md).
The spline domain is settled there: the default is the far image corner, the
model's own reach, and the outermost keypoint with its button is shown wherever
the domain is edited, so a circular fisheye is trimmed to its image circle by
choice.
