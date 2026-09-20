# The bench's active track in the 3D viewer

**Status:** Draft

The Image Detail panel draws the bench's active track in each photograph that
observes it, and every mark it draws is a handle
([`../gui/multi-panel-image-browser.md`](../gui/multi-panel-image-browser.md)
§ "The bench layer"). This draft proposes the same thing in the 3D viewer: the
track's patch frame drawn where it stands in the world, with its normal and with
one mark per observation saying where that photograph sees the patch's content,
and the same frame taken hold of there. Two of its handles have no counterpart
in a photograph, because a photograph cannot say them: moving the patch along
its normal, and tilting it.

It replaces the "ghost point" sketch in
[`sfm-explorer-track-editing.md`](sfm-explorer-track-editing.md) Part 7, and
ends the non-goal "Drawing the bench in the 3D viewer" in
[`../gui/bench.md`](../gui/bench.md).

---

## What is drawn

The layer draws **the active item of the active node's bench, when it is a
track-stage track**. A cluster-stage item has no geometry and draws nothing.
A track at infinity draws a reduced figure (§ "A track at infinity"). Items
that are not active draw nothing, as in Image Detail.

It is drawn **in the scene, by the GPU**, the way the target compass is
([`../gui/point-cloud-rendering.md`](../gui/point-cloud-rendering.md) § "Depth-Aware
Transparency"): a pass after the EDL pass that samples the scene's hardware
depth buffer, so each fragment knows whether it is in front of the scene's
geometry or behind it. The hardware buffer and not the linear-depth attachment,
which a patch, a frustum and an image quad all write as zero: a frame sunk into
a surfel has to read as behind it. Under the infinite reversed-Z projection
`ndc_z = near / view_depth`, so dividing the two depths back out states the gap
between them in world units, which is what lets the falloff below be in the
patch's own size. The figure therefore shows **how it meets the geometry around
it**: where the frame cuts through the point cloud or a patch, the part in
front is drawn at full strength and the part behind is drawn through, dimmed.

| Depth relationship | Drawn |
|---|---|
| In front of the scene | its colour, full opacity |
| Behind the scene | the same hue, opacity `0.45 * exp(-depth_behind / fog_distance)`, floored at `0.15` |

`fog_distance` is `4h`, two side lengths, so the falloff is in the patch's own
units: a frame sunk a fraction of its size into a surface reads as slightly
veiled, and one well behind it sits at the floor. The floor is what keeps every
handle findable, since a handle that cannot be seen cannot be grabbed.

Two things differ from the compass. The blend is **ordinary alpha, not
additive**, and there is no glow: the three violets carry meaning, and additive
light over a bright point cloud washes them all to white. And the hue does not
shift when occluded, for the same reason: a verdict's colour stays that
verdict's colour.

The pass reuses the compass's two pipelines' technique: **instanced
ribbon-quad edges** with homogeneous endpoints for every segment and circle
(the frame, the normal, the barbs, the marks' segments and hollow circles), and
a **triangle list** for the filled centre dot. Each instance carries its own
colour. Line width is in screen pixels, the Image Detail layer's stroke. The
geometry is rebuilt on the CPU whenever the track, the drag preview or the
viewer's eye (for the barbs) changes, which is a few hundred instances at most.
The three violets are the Image Detail layer's own constants, moved to
`crate::bench` so the two panels cannot drift.

**Hit-testing stays on the CPU**, against the same world geometry projected
through the viewer's camera: it needs no pick-buffer entry, and occlusion does
not affect it. A handle drawn through the cloud is grabbed like any other.

With `c` the frame's centre, `u` and `v` its unit axes, `h` its half-length and
`n = u x v` its outward normal:

- **The frame**: the square `c +/- h u +/- h v`, a closed polyline, in the `in`
  colour whatever any observation's verdict is. Each edge is clipped on its
  own, so a frame partly behind the viewer draws the edges that are in front.
- **The centre dot**: a filled disc **in the frame's plane**, radius `h / 8`,
  drawn as a filled polygon of 24 world points `c + r (cos t u + sin t v)`. It
  foreshortens with the frame and goes to a line when the frame is edge-on,
  which is what says it is part of the patch and not a billboard like a 3D
  point.
- **The normal**: the segment from `c` to `a = c + L n` with `L = 2h`, one side
  length, in the `in` colour, ending in an arrowhead. The arrowhead is two
  barbs from `a` back to `a - 0.25 L n +/- 0.1 L b`, where `b` is the unit
  vector along `n x (eye - a)`: perpendicular to the normal and as square to the
  viewer as it can be, so the head never goes edge-on. When `n` points at the
  eye that cross product vanishes and `b = u`.
- **One mark per observation**, all of them whatever their verdict, each in its
  own verdict's colour. The mark is the reverse of Image Detail's offset
  segment. The observation's keypoint is a ray from its camera; the ray meets
  the frame's plane at `q_i = c + a_i u + b_i v`, where `(a_i, b_i)` is
  `OrientedPatch::keypoint_plane_offset`. The mark is the segment `c -> q_i`
  and a **hollow circle in the plane** at `q_i`, radius `h / 8`, 24 segments.
  A track with 16 observations shows 16 segments and 16 circles. An
  observation with no keypoint, or whose ray does not meet the plane in front
  of its camera, draws no mark.

The marks all lie in one plane with the frame, so from the side the whole
figure collapses to a line with the normal standing out of it. That is correct
and is the view in which the normal handle is easiest to use.

**Everything in the frame's plane is drawn `h / 1000` off it**, along the
outward normal. The patch's own bitmap lies in that plane, and geometry exactly
coplanar with it would flicker between in front and behind. The lift is a
fraction of `h` so that it does not depend on the scene's size. A track at
infinity is not lifted.

**Sizes are world sizes.** The dot, the circles and the arrow scale with the
patch, so the figure looks the same at every distance and is small when the
patch is small on screen. Handles are reached by zooming in, as in Image
Detail.

### What the panel is told

As with Image Detail, the viewer does not read the bench. The dock hands it one
value per frame: the active node's id, the active track, the
`EditedReconstruction` at the cursor for the camera poses the marks need, the
node's similarity, the observation row selected in Track Edit, and whether a
task holds the node. While a task holds the node the layer draws and no handle
takes a press. The selected row's circle is drawn 1.6 times the others', which
is the other half of the click that sets it: a row picked in Track Edit can be
found out in the world, and a mark picked in the world can be seen to be that
row.

### A track at infinity

A `w = 0` frame is a **direction patch**: `c` is a unit bearing `d`, the frame
is tangent to the unit sphere at `d`, and its corners `d +/- h u +/- h v` are
themselves directions, `h` being a tangent length. It is drawn the way the
viewer draws a point at infinity
([`../gui/point-cloud-rendering.md`](../gui/point-cloud-rendering.md) § "Points
at Infinity"): every direction is projected **rotation-only**, with the viewer's
translation left out, so the figure has no parallax and stays on the sky as the
viewer moves. The ribbon instances carry `w = 0` endpoints, which is that
projection. A direction is behind every finite thing, so wherever the scene has
depth the figure is drawn through it, at the floor opacity with no falloff, and
at full strength against the background.

- **The frame**, the **centre dot** and **one mark per observation** are drawn
  exactly as above with directions in place of points. An observation's mark is
  its keypoint's ray **direction** taken into the world by its camera's rotation
  alone and met with the tangent plane: `q_i = r_i / (r_i . d)`, which is `d +
  a_i u + b_i v`. A ray with `r_i . d <= 0` draws no mark.
- **There is no normal.** A direction patch's normal is fixed by its bearing,
  so there is nothing to say and nothing to grab: no segment, no arrowhead.
- A frame whose bearing is behind the viewer draws nothing.

---

## The handles

The rules of Image Detail's handles hold here unchanged, and this section only
states what differs:

- **The press decides which handle, not the drag.** A primary press in the
  viewport is hit-tested against the figure as drawn that frame. If it lands on
  a handle the gesture is that handle's until the button comes up, and the
  orbit, pan and zoom that a drag would otherwise drive are suppressed from the
  press. A press that hits nothing leaves navigation exactly as it is. Only an
  unmodified primary press is tested: Alt, Ctrl and Shift drags and the middle
  and secondary buttons stay navigation's.
- **One version per drag**, through `AppState::edit_bench_patch`. While the
  button is down the layer draws the track `bench::geometry::apply` would
  produce from the pointer where it is. **Escape abandons the drag.** A drag
  that ends where it started pushes nothing. A press that never moves is a
  click: on an observation's circle it selects that row in Track Edit, anywhere
  else on the figure it does nothing.
- **Reach**: nine panel pixels for the dot, a corner, the arrowhead and a
  circle; eight from an edge or from the normal's segment. Where reaches
  overlap the order is arrowhead, corner, dot, circle, edge, normal segment.
  The arrowhead is first because it is a point at the far end of the segment
  that is last, and anything between them would take the presses meant for it.
  One consequence is worth knowing: seen exactly down the normal the whole
  arrow collapses onto the centre, so the head covers the dot and takes its
  presses. That view is the aim's own best one, and a few degrees of lean pulls
  the head clear and gives the dot back.
- **Cursors**: `Move` on the dot and `Grabbing` while it is held, and a circle
  selects rather than moves, so it takes `PointingHand`. The normal's segment
  takes the resize cursor along its own on-screen direction. The arrowhead takes
  the cursor of the gesture it is about to make (§ "The arrowhead's two
  gestures"): `AllScroll` where the aim is free in two directions, and where it
  is a swing about one axis, the resize cursor along the arc the arrowhead
  travels, which is the corner's own reading of a tangent.

  An edge and a corner both take the resize cursor their on-screen orientation
  names, and the difference between them is the direction: an edge is dragged
  **across** itself, so its cursor is the perpendicular of the edge; a corner
  travels **along** the arc it turns on, so its cursor is the perpendicular of
  its own radius from the centre, which is that arc's tangent. Running the
  pointer down an edge and onto the corner turns the cursor by the difference
  between the two gestures, which is what says one resizes and the other
  rotates. There is no rotation cursor to give a corner instead: egui's set is
  the CSS one, and neither has ever had such a thing. The arc glyph Image Detail
  draws beside a hovered corner has no counterpart here, every mark of this
  figure being scene geometry drawn by the pass where a hint that follows the
  pointer is not.

**The pointer is a ray of the viewer's camera**, where in Image Detail it is a
ray of a reconstruction camera. Everything else about reading it is the same
idea: the ray is met with the patch's own geometry, and the edit is stated in
the patch's own terms. The ray is taken back through the node's similarity
first, because the figure is drawn in the world and the core steps act in the
reconstruction's own coordinates; a similarity preserves angles, so the
degenerate-view tests read the same on either side of it.

| Handle | The pointer ray is met with | The edit |
|---|---|---|
| Centre dot | the frame's plane | slide the centre to that point, less the press's own offset from the centre |
| Edge | the frame's plane | the offset along that edge's axis is `p`; new half-length `(p + h) / 2`, centre moved `h' - h` along it, far edge held |
| Corner | the frame's plane | the angle swept about `n` from the press point to the pointer, both read about `c` |
| Normal segment | the line `c + t n` | the point of that line nearest the ray; the centre moves by that point's `t` less the press's own, along `n` |
| Arrowhead, aiming | the plane through `c` square to `n` | the new normal is `4h n` carried by the pointer's travel across that plane since the press, normalized |
| Arrowhead, swinging | the plane through `c` square to the swing axis `a` | the angle swept about `a` from the press's point to the pointer's turns the normal about `a` |

**The degenerate views are refused at the press.** Three handles read the
frame's plane, and a plane seen edge-on turns a pixel of pointer motion into an
unbounded distance. When the angle between the view ray through `c` and the
plane is under 5 degrees, the dot, the edges and the corners take no press and
show the default cursor. The normal segment is the mirror case: when the angle
between `n` and the view ray is under 5 degrees the nearest-point solve is
ill-conditioned and the segment takes no press. The **line** is undirected, so
that angle is read as a magnitude: a view straight up the normal is exactly as
bad as one straight down it. Both tests are the same cosine, one refusing the
view where it goes to zero and the other the view where it goes to one, which
is what makes them complementary rather than merely alike: the nearest-point
solve divides by `1 - (d . n)^2`, the square of the sine the plane's own
reading is multiplied by. Between them some handle is
always live, and the view in which one set dies is the view in which the other
is at its best.

The arrowhead needs no refusal of its own, because its two gestures are each
other's cure: the aim reads a plane that is square to the view exactly when the
normal points along it, and the swing reads an axis that is well determined
exactly when it does not. Both planes run through `c`, so neither can fall
behind an eye that is looking at the figure at all, and the arrowhead answers
from every view it is drawn in.

**The arrowhead's two gestures.** Which one a press makes is decided by where
the normal points. With `e` the unit vector from `c` to the eye, the arrowhead
**aims** when `|n . e|` is above `cos 45 degrees` and **swings** when it is
below. The magnitude and not the signed value, because a patch showing its back
is as square to the view as one showing its face. The press decides it, as the
press decides the handle, so a gesture does not change character halfway
through.

**Aiming**, when the normal lies near the line of sight. The pointer's ray is met
with the plane through `c` square to `n`, and the new normal is `4h n` -- the old
normal on a lever twice the arrow's own length -- displaced by however far the
pointer has travelled across that plane since the press, normalized. The plane
and the lever are the ones fixed at the press, `n` being the normal the arrowhead
had then, so the gesture is a single map from the window onto the sphere of
normals rather than a thing that moves as it is used. The press's own place is
kept for the reason the centre dot's is: a press that took the arrowhead is not
standing where `c` projects, the head being drawn out along the normal, and
reading the meeting outright would turn the normal over before the pointer had
moved at all -- which a drag that ends where it started is not allowed to do.

Two properties follow from the lever. `4h` of travel is 45 degrees of tilt while
the arrowhead is drawn `2h` out, and the travel is read on the very plane the
arrow stands out of, so `4h` is **twice the arrow's own drawn length whatever the
zoom**: the aim is half as sensitive as the figure looks, at every distance, and
a small correction is a small motion, which is what the handle is for -- a normal
is read off a surface a few degrees at a time. And the travel being square to
`n`, the answer keeps the whole `4h` along it; the sum of a fixed vector and one
square to it can never turn through a right angle, so one gesture turns the
normal by less than 90 degrees and cannot push it through the frame at all.

**The lever is not a distance the plane stands at**, and that distinction is the
whole of why the plane runs through `c`. Standing the plane `4h` out along `n`
gives the same arithmetic wherever the eye is far off, and fails where it is not:
the standoff is measured *toward* an eye the aim was chosen for, so an eye within
`4h` of a patch facing it has the plane behind it, the ray meets nothing, and the
press falls through to the viewport's navigation with the arrowhead's cursor
still showing. That view -- close in on a patch that faces you -- is not a corner
case but the one a person zooms to when they mean to work on a normal. Reading
`c`'s own plane also takes the eye's distance out of the gesture entirely: what a
pixel of pointer is worth on a plane depends on how far off that plane is, so a
plane standing at a fixed offset makes the handle's sensitivity a function of the
zoom, while `c`'s own plane and the patch project alike and the ratio between
them is the same at every zoom.

**Swinging**, when the normal lies across the line of sight. The normal turns
about one axis `a`, which is **in the frame's own plane**: the unit vector there
nearest the eye, which is the part of `e` square to `n`, normalized. The normal
therefore stays in the one plane through it square to `a` and never rolls toward
or away from the viewer, which is the motion that is hardest to aim and easiest
to overshoot when the arrowhead is nearly side-on. The pointer's ray is met with
the plane through `c` square to `a`, which is the plane the arrowhead travels in
and, `a` pointing at the eye as nearly as the frame allows, the plane most nearly
facing the window; the angle swept about `c` from the press's meeting to the
pointer's is the angle the normal turns by. That is the corner's reading with `a`
in place of `n`, which is why the two take the same cursor.

Both state their answer as a normal, so `tilt_frame` takes a normal and knows
nothing of which gesture named it. Both turn the frame by the **least rotation** that takes
the old normal to the new one, the rotation about `n_old x n_new`, so a tilt adds
no spin about the normal: spin is the corners' job. A swing already is that
rotation, `a` being square to both normals by construction.

**At infinity** the pointer is its ray's **direction** alone, met with the
tangent plane as an observation's is: `r / (r . d)`. The dot, the edges and the
corners work from that point exactly as they do for a finite frame, through the
same core steps, which already carry a `w = 0` frame: a slide moves the bearing
and renormalises it, a resize changes the tangent half-length, a turn spins the
square about `d`. The normal segment and the arrowhead do not exist. The plane
is never edge-on, since the viewer is in effect at the sphere's centre, so the
one refusal is a pointer more than 85 degrees from `d`, where `r . d` goes to
zero. `offset_frame` and `tilt_frame` refuse a `w = 0` track in their own
sentence, which the wire tools pass on.

### What each edit does to the sightings

The slide, the resize and the turn are the steps Image Detail already makes and
keep their rules: a move of the centre carries every keypoint along the plane by
the same displacement, a turn moves none, nothing is pinned, the measurements
and the consensus bitmap are dropped.

The two new edits follow the same principle, which is that **a keypoint's
in-plane offset `(a_i, b_i)` is the thing kept**: it is where that photograph
sees the patch's content against where the geometry puts its middle, and it is
what the tiles are cut on.

- **Offset along the normal.** `c' = c + d n`. Each observation's keypoint
  becomes the projection of `c' + a_i u + b_i v` into its image. An observation
  the moved patch no longer projects into is left with no keypoint and
  `Unmeasured::NoProjection`, as a slide leaves it. `(a_i, b_i)` is read on the
  plane the sightings were measured against, which is the plane the patch is
  **leaving**: unlike a slide, this edit takes the plane with it, so a reading
  taken against the plane already reached would displace every sighting a
  second time. The sightings therefore move by *different* amounts in their
  photographs, and that spread is the parallax the old depth was wrong by.
- **Tilt.** `u`, `v` and so `n` are rotated about `c`. Each keypoint becomes
  the projection of `c + a_i u' + b_i v'`. `(a_i, b_i)` is read on the frame as
  it stood **before** the turn, for the reason the offset's is: this edit takes
  the plane with it too. Unlike the other three this is not a rigid carry of
  each plane point -- the pair is kept and the place is built again on the new
  axes -- but what it preserves is the same thing.

**A tilt stops 80 degrees from any observation.** With `e_i` the unit vector
from `c` to observation `i`'s camera centre, a normal is **allowed** when the
angle between it and `e_i` is at most 80 degrees for every observation, whatever
its verdict. The allowed normals are the intersection of one spherical cap per
observation, so dragging the arrowhead against the limit traces the edge of what
the existing observations can see, which is the point: the person feels where
the normal can be. The step takes the requested normal, walks the great arc from
the current normal toward it, and stops at the last allowed normal on that arc.
An observation already past 80 degrees before the tilt constrains nothing, since
otherwise a track that starts outside the region could not be tilted back into
it. The rule is `tilt_frame`'s own, so the drag and the wire tool stop at the
same place, and a stopped tilt says so as a clamped pixel does: the report
carries the normal asked for, and the label ends *"stopped 80.0 degrees from
IMG_0042.jpg"*.

Both drop the measurements and the bitmap, and pin nothing.

---

## The core and viewer interface

Two things in `sfmtool_core::bench` change shape, and two steps are added.

**The slide and the resize get world-point forms.** `translate_frame` and
`resize_from_edge` take a pixel of an observation's image, unproject it onto the
plane, and act. The part after the unprojection becomes the step, taking the
point on the plane, and the pixel form becomes the unprojection in front of it.
There is one implementation of each edit and both panels reach it. The
unprojection is the same one for both: the pointer's offset is read from the
**outline's** centre -- the surfel re-anchored on that sighting -- and carried to
the surfel's own centre, and that is the place the step is handed.

The slide's report is `TranslateToReport` rather than `TranslateFrameReport`:
the latter names an observation, an image and a pixel, and a caller naming a
place in the world has none of the three. `ResizeReport` already carries its
three as `Option`, `resize_frame` naming no pixel either, so the resize reuses
it with all three `None`. A place that is not a finite point is refused as
`TrackEditError::BadPlace`, the way a pixel that is not one is refused as
`BadPixel`.

```rust
/// Slide the surfel until its centre sits at `point`, which is projected onto
/// the frame's plane first. Every sighting follows.
pub fn translate_frame_to(track: &EditableTrack, edited: &EditedReconstruction,
                          point: Point3<f64>)
    -> Result<(EditableTrack, TranslateToReport), TrackEditError>;

/// Put `edge` at `point`'s offset along that edge's axis, the far edge held.
pub fn resize_from_edge_to(track: &EditableTrack, edited: &EditedReconstruction,
                           edge: Edge, point: Point3<f64>)
    -> Result<(EditableTrack, ResizeReport), TrackEditError>;

/// Move the surfel `distance` world units along its outward normal.
pub fn offset_frame(track: &EditableTrack, edited: &EditedReconstruction,
                    distance: f64)
    -> Result<(EditableTrack, OffsetFrameReport), TrackEditError>;

/// Turn the surfel about its centre, by the least rotation, toward the outward
/// normal `normal`, stopping 80 degrees from any observation's camera.
pub fn tilt_frame(track: &EditableTrack, edited: &EditedReconstruction,
                  normal: Vector3<f64>)
    -> Result<(EditableTrack, TiltFrameReport), TrackEditError>;
```

One difference from Image Detail is deliberate. There the outline is the surfel
**re-anchored on that image's keypoint**, because the panel shows where the
sighting is in that photograph. Here there is no photograph and the frame drawn
and grabbed is the surfel itself, at `c`.

In the viewer, `PatchEdit` gains four variants, each one `apply` arm:

```rust
SlideTo { point: [f64; 3] },
ResizeFromEdgeTo { edge: Edge, point: [f64; 3] },
Offset { distance: f64 },
Tilt { normal: [f64; 3] },
```

`Rotate { angle_rad }` is used as it stands. The no-effect tolerances follow the
existing rule, the units of the value moved: an offset within a millionth of the
half-length, a tilt within a nanoradian.

**Version labels**, beside the existing three:

| Step | Version label |
|---|---|
| Slide to a place | `Moved pt3d_a1b2c3d4_1207 by 0.042 units to (1.204, -0.318, 4.006)` |
| Resize to a place | `Resized pt3d_a1b2c3d4_1207 to a half-length of 0.0184` |
| Offset along the normal | `Moved pt3d_a1b2c3d4_1207 by 0.042 units along its normal to (1.204, -0.318, 4.006)` |
| Tilt | `Tilted pt3d_a1b2c3d4_1207 by 8.4 degrees` |

The first two are the sentences the pixel forms already write, minus what a
photograph gave them. A slide's loses the clamp note, no pixel having been
brought inside a picture. A resize's reports the **world half-length**: the
existing sentence names a size in the pixels of the sighting the gesture came
through, and a gesture out in the world came through none, so it falls back to
the world number the label already carries for a patch that does not project
into its sighting's image.

**The wire** gets a tool per new edit, each one `edit_bench_patch`, so a drag
and a call are the same version with the same sentence:

```jsonc
// offset_bench_track { "reconstruction_label": "bull", "distance": 0.042 }
// tilt_bench_track   { "reconstruction_label": "bull", "normal": [0.1, -0.2, 0.97] }
```

The slide and the resize get no second wire form: `move_bench_track` and
`resize_bench_track` already say them, by pixel.

The figure's world geometry and the handles are a new
`viewer_3d/bench_track.rs`; the pass is `scene_renderer/bench_track.rs` with
`shaders/bench_track.wgsl`; the ray geometry (ray against a
plane and against a line, and the degenerate-view tests) goes in
`bench/geometry.rs` beside what a pixel means against a patch, with no egui in
it, so it is tested without a frame.

---

## Implementation steps

Each is one PR, and each leaves the viewer whole.

1. **Rendering.** The value the dock hands the viewer, the figure's world
   geometry, the GPU pass and its shader, the shared colour constants. No core
   change, no input. Tests, on the `noop` backend as
   `scene_renderer/upload/tests.rs` does and on the geometry builder directly:
   the instance list holds the frame, the dot, the arrow and one mark per
   observation with a keypoint; a track at infinity holds its frame, dot and
   marks with `w = 0` endpoints and no normal;
   a cluster-stage active item uploads nothing; the
   mark's `q_i` reprojects onto the keypoint it came from. The depth-aware
   blend is checked by eye with MCP screenshots of the viewport: a frame lying
   on a surface, one in front of it and one behind it.
2. **The three handles Image Detail has**: dot, edges, corners. The two
   world-point core steps with the pixel forms rebuilt on them, the two
   `PatchEdit` variants, the press rule and navigation's suppression, the
   edge-on refusal, Escape, click-selects-row. Tests drive real frames as
   `image_detail/tests.rs` does: a press on an edge and a sub-threshold motion
   resizes and orbits nothing, while the same motion from empty viewport orbits;
   the dragged edge reprojects under the release point with the far edge held; a
   corner dragged onto its neighbour is a quarter turn; one version per drag;
   Escape leaves nothing; an edge-on view takes no press; the same three
   handles on a track at infinity move the bearing, the tangent half-length and
   the spin.
3. **The normal segment.** `offset_frame`, `PatchEdit::Offset`,
   `offset_bench_track`, the down-the-normal refusal, the `w = 0` refusal. Tests: the centre lands on
   the normal line under the pointer; every `(a_i, b_i)` is unchanged and every
   keypoint is the new projection; an observation that loses its projection
   carries `NoProjection`.
4. **The arrowhead.** `tilt_frame`, `PatchEdit::Tilt`, `tilt_bench_track`, the
   aim, the swing and the rule that chooses between them. Tests: the tilt is the
   least rotation (the axis is perpendicular to both normals and `u` keeps its
   component along it); the centre and the half-length are unchanged; a tilt
   asked past an observation stops 80 degrees from it and says which, while a
   track already past that for one observation can still be tilted; a pointer
   travelling `4h` across the aim's plane is 45 degrees and no aim reaches 90;
   an aim answers from an eye closer in than its own lever, where a plane stood
   off by that lever would sit behind the camera; a swing leaves the normal
   square to its axis and the axis in the frame's plane; the gesture is chosen
   at the press and does not change while the button is down.
5. **The names.** The steps and the wire tools are named for the shape of the
   arithmetic rather than for what a person does with them, and they collide:
   `translate_frame`, `translate_frame_to`, `offset_frame` and
   `resize_from_edge_to` all move a centre, and `move_bench_track`,
   `move_bench_track_observation` and `offset_bench_track` all read as the same
   verb. One pass settles a vocabulary and renames the core steps, their reports,
   the `PatchEdit` variants and the wire tools to it, rewriting the standing
   specs in the same change. No behaviour moves, so the tests are the ones
   already written, renamed with what they call. The old wire names are not kept
   as aliases: an agent binds its tools when a session starts, so there is one
   generation of callers to move, and a shim would make the ambiguous name
   permanent.

On step 4 landing this draft is converted: the panel half into
`../gui/viewer-3d-bench-layer.md` or a section of the 3D viewer's spec, the four
steps into [`../core/bench/editable-track.md`](../core/bench/editable-track.md),
the tools into [`../gui/bench.md`](../gui/bench.md) § "The wire" and
[`../gui/mcp-server.md`](../gui/mcp-server.md).

---

## Decided in review

- An edge only resizes; the centre dot is what slides the patch.
- The arrow is `2h` long in world units, to be looked at and adjusted once it is
  on screen.
- Camera view mode needs nothing: the 3D view is always an undistorted
  perspective projection, and the distorted photograph behind it is what gets
  warped, onto a distorted grid mesh. The layer is scene geometry and lines up
  as the point cloud does.
- A tilt stops 80 degrees from any observation (§ "What each edit does to the
  sightings").
- The arrowhead has two gestures chosen by where the normal points, rather than
  one reading of a sphere about `c` (§ "The arrowhead's two gestures"). A sphere
  makes every drag a full-sensitivity aim, including the side-on view where the
  useful motion is a swing about a single axis and the rest is roll the person
  did not ask for; and a sphere of the arrow's own radius turns 90 degrees in the
  length of the arrow, which is far too fast for a handle whose job is a few
  degrees at a time.
- The aim's `4h` is a **lever** and not a distance its plane stands at: the
  plane runs through `c`, and the old normal is carried on `4h` of lever, so the
  gesture is half as sensitive as the figure looks. It is a number to look at on
  screen and adjust, as the arrow's own `2h` is. Standing the plane off by the
  lever instead reads the same wherever the eye is far away and fails where it
  is not, the standoff being measured toward an eye the aim was chosen for: from
  inside `4h` the plane lies behind the camera and the press falls through to
  navigation while the cursor still says otherwise. Reading `c`'s own plane also
  takes the eye's distance out of the sensitivity, which a standoff puts into
  it.
