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
value per frame: the active node's id, the active track (an
`Arc<EditableTrack>`), the `EditedReconstruction` at the cursor for the camera
poses the marks need, the observation row selected in Track Edit, and whether a
task holds the node. While a task holds the node the layer draws and no handle
takes a press.

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
- **Cursors**: `Move` on the dot, the on-screen-orientation resize cursor on an
  edge, `Alias` with the small arc on a corner, as in Image Detail. The normal's
  segment takes the resize cursor along its own on-screen direction. The
  arrowhead takes `Grab`, and `Grabbing` while held.

**The pointer is a ray of the viewer's camera**, where in Image Detail it is a
ray of a reconstruction camera. Everything else about reading it is the same
idea: the ray is met with the patch's own geometry, and the edit is stated in
the patch's own terms.

| Handle | The pointer ray is met with | The edit |
|---|---|---|
| Centre dot | the frame's plane | slide the centre to that point, less the press's own offset from the centre |
| Edge | the frame's plane | the offset along that edge's axis is `p`; new half-length `(p + h) / 2`, centre moved `h' - h` along it, far edge held |
| Corner | the frame's plane | the angle swept about `n` from the press point to the pointer, both read about `c` |
| Normal segment | the line `c + t n` | the point of that line nearest the ray, less the press's own `t`; the centre moves there |
| Arrowhead | the sphere of radius `L` about `c` | the new normal is the unit vector from `c` to the hit |

**The degenerate views are refused at the press.** Three handles read the
frame's plane, and a plane seen edge-on turns a pixel of pointer motion into an
unbounded distance. When the angle between the view ray through `c` and the
plane is under 5 degrees, the dot, the edges and the corners take no press and
show the default cursor. The normal segment is the mirror case: when the angle
between `n` and the view ray is under 5 degrees the nearest-point solve is
ill-conditioned and the segment takes no press. Between them some handle is
always live, and the view in which one set dies is the view in which the other
is at its best.

**The arrowhead's sphere.** The ray meets the sphere twice, or not at all. The
hit taken is the one on the hemisphere the arrowhead was on at the press, near
or far from the viewer, so a drag does not flip the normal through the frame.
When the ray misses the sphere the hit is the point of the sphere's silhouette
nearest the ray, so the arrowhead follows the pointer around the rim. The frame
is turned by the **least rotation** that takes the old normal to the new one,
the rotation about `n_old x n_new`, so a tilt adds no spin about the normal:
spin is the corners' job.

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
  `Unmeasured::NoProjection`, as a slide leaves it.
- **Tilt.** `u`, `v` and so `n` are rotated about `c`. Each keypoint becomes
  the projection of `c + a_i u' + b_i v'`.

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
There is one implementation of each edit and both panels reach it.

```rust
/// Slide the surfel until its centre sits at `point`, which is projected onto
/// the frame's plane first. Every sighting follows.
pub fn translate_frame_to(track: &EditableTrack, edited: &EditedReconstruction,
                          point: Point3<f64>)
    -> Result<(EditableTrack, TranslateFrameReport), TrackEditError>;

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
| Offset along the normal | `Moved pt3d_a1b2c3d4_1207 by 0.042 units along its normal to (1.204, -0.318, 4.006)` |
| Tilt | `Tilted pt3d_a1b2c3d4_1207 by 8.4 degrees` |

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
`shaders/bench_track.wgsl`; the ray geometry (ray against
plane, line and sphere, and the degenerate-view tests) goes in
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
   hemisphere and silhouette rules. Tests: the tilt is the least rotation (the
   axis is perpendicular to both normals and `u` keeps its component along it);
   the centre and the half-length are unchanged; a tilt asked past an observation
   stops 80 degrees from it and says which, while a track already past that for
   one observation can still be tilted; a pointer off the sphere tracks
   the rim; a drag across the silhouette does not flip the normal.

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
