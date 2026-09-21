# The bench's active track in the 3D viewer

The SfM Explorer's **bench** is where a single 3D point is held and worked on by
hand before it is written into a reconstruction: its observations across the
photographs, and the oriented square of surface those observations are views of.
The 3D viewer draws the track that is active on that bench where it stands in
the world, as scene geometry rather than as a symbol floating over the viewport.
What it draws is the square itself, the direction that square faces, and one
mark per photograph saying where that photograph sees the square's content
against where the geometry puts its middle. Every one of those marks is also a
handle, so the figure a person is looking at is the figure they take hold of and
there is no second picture of the patch to keep in step with the first.

The Image Detail panel already draws the same track in each photograph that
observes it, with the same marks acting as handles
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
layer"). What the world view adds is the two gestures no photograph can express.
A picture fixes the ray a piece of surface lies along; it is silent about how
far down that ray the surface is, and silent about which way the surface faces.
Those two are settled out here, on the normal's own segment and on the arrowhead
at the end of it.

## The layer, and what it is handed

The figure's world geometry and its handles are
[viewer_3d/bench_track.rs](../../crates/sfm-explorer/src/viewer_3d/bench_track.rs);
the GPU pass that draws them is
[scene_renderer/bench_track.rs](../../crates/sfm-explorer/src/scene_renderer/bench_track.rs)
with
[shaders/bench_track.wgsl](../../crates/sfm-explorer/src/shaders/bench_track.wgsl);
and what a pointer means against a patch is
[bench/geometry.rs](../../crates/sfm-explorer/src/bench/geometry.rs), which the
Image Detail handles and the wire's patch tools read a pixel through as well.
The edits themselves are the core steps in
[bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs), described in
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Placing,
sizing and turning by hand". Nothing in the geometry module holds a window or a
device, and the ray arithmetic carries no egui, so the figure can be asserted
without a device and a pointer reading without a window.

The viewer does not read the bench, for the reason the Image Detail panel does
not: the panel is handed `&mut` into the state further down the same call, so
what it needs is read out beside the selection and passed in. The dock hands the
viewport one value per frame, carrying the node whose bench it is, the active
track, the node's edited reconstruction at its cursor (for the camera poses the
marks unproject their keypoints through), the node's similarity, the observation
row Track Edit has selected, and whether a background task holds the node. A
task holding the node does not stop the layer drawing, because what is being
worked on does not stop being worth seeing; it stops every handle taking a
press.

**The layer draws the active item, and only when it is a track-stage track.** A
cluster-stage item has no shared geometry behind it and draws nothing, as does a
track nothing has given a patch. Items that are not active draw nothing, which
is what Image Detail does too: the bench holds several and both panels show the
one being worked on.

**While a handle is held, what is drawn is the preview.** The figure is rebuilt
from the track the release would push, through the same call the release goes
through, so there is one answer rather than a drawn guess beside a pushed
result. The figure is also rebuilt after the frame's input has moved the camera,
because the arrowhead's barbs are squared to the eye.

## What is drawn

Write `c` for the patch's centre, `u` and `v` for its unit axes, `h` for its
half-length and `n = u x v` for its outward normal. A patch is square, so one
half-length states the whole of its size.

- **The square** `c +/- h u +/- h v`, drawn as four edges walked in the order
  [`OrientedPatch::boundary`](../../crates/sfmtool-core/src/patch/cloud.rs)
  walks its corners. It is drawn in the `in` colour whatever any observation's
  verdict is: the square is the patch, not anybody's opinion of it. Each edge is
  clipped on its own against the near plane, so a square partly behind the
  viewer draws the edges that are in front of it.
- **The centre dot**, a filled disc lying *in the patch's plane*, of radius
  `h / 8`, built as a fan of 24 triangles about `c`. Being in the plane it
  foreshortens with the square and collapses to a line when the square is seen
  edge-on, which is what says it is part of the patch rather than a billboard
  like a 3D point splat.
- **The normal**, a segment from `c` out to the tip `a = c + 2h n`, one side
  length, ending in an arrowhead of two barbs. Each barb runs from `a` back to
  `a - 0.5h n +/- 0.2h b`, where `b` is the unit vector along `n x (eye - a)`:
  perpendicular to the normal and as square to the viewer as it can be, so the
  head never goes edge-on. When the normal points straight at the eye that cross
  product vanishes and the patch's own `u` stands in.
- **One mark per observation**, all of them whatever their verdict, each in that
  observation's own verdict colour. An observation already judged *in* is
  exactly the one whose answer is worth having in view. The mark is the reverse
  of Image Detail's offset segment: there the patch's projection is the hollow
  circle and the keypoint the dot, here the patch's centre is the dot and the
  keypoint the circle. The observation's keypoint is a ray from its camera, and
  that ray names a place `q_i` on the patch's plane through
  `OrientedPatch::keypoint_plane_offset`, the same unprojection every other
  reading of a keypoint against a patch goes through. The mark is the segment
  `c -> q_i` plus a hollow circle at `q_i`, of the same radius `h / 8` and the
  same 24 segments, drawn in the plane. A track with sixteen observations shows
  sixteen segments and sixteen circles. An observation with no keypoint, or one
  whose ray cannot meet the plane in front of its own camera, draws no mark at
  all and leaves a gap in the list rather than a mark that means nothing.

The circle of the observation Track Edit has selected is drawn 1.6 times the
others'. That is the one thing the figure says about the selection, and it is
the other half of the click that sets it: a row picked in the panel can be found
out in the world, and a mark picked in the world can be seen to be that row.

Because every mark lies in one plane with the square, the whole figure collapses
to a line from the side, with the normal standing out of it. That is correct,
and it is the view in which the normal's own handle is easiest to use.

**Everything drawn in the patch's plane is lifted `h / 1000` along the outward
normal.** The patch's own bitmap lies in that plane, and the pass reads
in-front-or-behind off the depth buffer, so geometry exactly coplanar with it
flickers between the two. Stating the lift as a fraction of `h` keeps it
independent of the scene's size. The arrow is not lifted, standing out of the
plane already, and a track at infinity is not lifted either.

**Sizes are world sizes.** The dot, the circles and the arrow all scale with the
patch, so the figure looks the same at every distance and is small on screen
when the patch is small. Handles are reached by zooming in, as they are in Image
Detail. The one measurement in pixels is the stroke: half-width 1.25 px, which
is the 2.5 px stroke the Image Detail bench layer draws the same track with, so
the two read as one layer.

### A track at infinity

A patch with `w = 0` is a **direction patch**. Its centre is a unit bearing `d`,
its square is tangent to the unit sphere at `d`, and its corners
`d +/- h u +/- h v` are themselves directions with `h` a tangent length. It is
drawn the way the viewer draws any point at infinity
([`point-cloud-rendering.md`](point-cloud-rendering.md) § "Points at Infinity"):
every direction is projected **rotation-only**, the viewer's translation
dropping out with the `w`, so the figure has no parallax and holds its place on
the sky as the viewer moves. Every endpoint the layer emits carries its own `w`,
and that `w` is the projection the shader gives it.

The square, the centre dot and the marks are all built exactly as above with
directions in place of places, an observation's `q_i` being where its ray's
direction meets the tangent plane. A ray pointing away from the bearing names no
such place and draws no mark. **There is no normal**: a direction patch's normal
is its own bearing, fixed by it, so there is nothing to say and nothing to grab,
and neither the segment nor the arrowhead is drawn. A figure whose bearing is
behind the viewer is clipped away entirely.

## Depth against the scene

The figure is drawn **in the scene, by the GPU**, in a pass that runs after the
EDL resolve and after the target indicator and the track rays, so it sits on top
of everything else the frame has drawn. It is the thing being worked on. Like
those two it *samples* the shared depth buffer rather than testing against it,
which is what lets a fragment behind the scene still be drawn, dimmed, instead
of discarded ([`point-cloud-rendering.md`](point-cloud-rendering.md) §
"Depth-Aware Transparency"). The figure therefore shows how it meets the
geometry around it: where the square cuts through the point cloud or through a
rendered patch, the part in front reads at full strength and the part behind is
drawn through.

| Depth relationship | Drawn |
|---|---|
| In front of the scene, or nothing there | its colour, full opacity |
| Behind the scene | the same hue at `0.45 * exp(-behind / fog_distance)`, floored at `0.15` |

It is the **hardware** depth buffer and not the linear-depth attachment. A
patch, a frustum and an image quad all write zero to the linear attachment, and
a square sunk into the surface a rendered patch draws has to read as behind it.
Under the infinite reversed-Z projection `ndc_z = near / view_depth`, so
dividing the two stored depths back out recovers the gap between them in world
units; that is what lets the falloff be expressed in the patch's own size:
`fog_distance` is `4h`, two side lengths, taken through the node's similarity
scale. A square sunk a fraction of its size into a surface reads as slightly
veiled, and one well behind it sits at the floor. The floor is what keeps every
handle findable, since a handle that cannot be seen cannot be grabbed.

A direction has no distance behind anything to fade over, so a `w = 0` figure is
drawn at the floor wherever the scene has depth at all and at full strength
against the empty background. Its fragments are pinned at a tiny positive NDC
depth, the same constant `points.wgsl` and `patch.wgsl` pin their own infinity
geometry at, which puts a direction just in front of the reversed-Z far plane
and behind every finite thing.

Two things differ from the target indicator this pass otherwise follows. The
blend is **ordinary alpha rather than additive**, and there is no glow: the
three bench violets carry meaning, and additive light over a bright point cloud
washes them all toward white. And the hue does not shift when the figure is
occluded, for the same reason. A verdict's colour stays that verdict's colour,
and only the opacity says the figure is behind something.

The pass borrows the indicator's two pipelines. Every straight piece of the
figure (the square's edges, the normal and its barbs, each mark's segment, and
each chord of each hollow circle) is one instance of a camera-facing **ribbon
quad** carrying its two homogeneous endpoints and its own colour, and the filled
centre disc is a **triangle list**. A triangle cannot be clipped a vertex at a
time the way a segment can, so a disc vertex behind the near plane drops its
whole triangle; the disc is an eighth of a half-length across, so its three
corners agree in every view but a grazing one. The figure is rebuilt on the CPU
and re-uploaded every frame it is drawn, since the barbs turn to face the eye
and there is no state to key it on that a camera move does not invalidate. It is
a few hundred instances.

## The handles

Every rule the Image Detail handles follow holds here, and this section says
only what the world view adds or does differently.

**The press decides which handle, not the drag.** A primary press in the
viewport is hit-tested against the figure as the last frame built it, which is
the figure on screen and so the one the person pressed on. If it lands on a
handle, the gesture is that handle's until the button comes up, and the orbit,
pan and zoom a drag would otherwise drive are suppressed from the press onward
rather than from the moment egui would call the motion a drag. An easing camera
transition is cancelled at the same moment, since the drag is a statement about
where the pointer is on the figure as it stands. A press that hits nothing
leaves navigation exactly as it was. Only an unmodified primary press is tested:
Alt, Ctrl, Cmd and Shift drags, and the middle and secondary buttons, are
navigation's and stay navigation's.

**One version per drag**, pushed through `AppState::edit_bench_patch`, which is
the call the wire's patch tools make ([`bench.md`](bench.md) § "The wire"), so a
drag out in the world and a tool call are the same version carrying the same
sentence, one Action Log row and one Undo. Nothing is pushed while the button is
down. **Escape abandons the drag**: the preview goes and nothing is pushed,
though the viewport still does not orbit until the button comes up, because what
the pointer means was decided where it went down. A drag that ends where it
started pushes nothing, the way a verdict an observation already holds does. A
press that never moves is a click, which on an observation's circle selects that
row in Track Edit and anywhere else on the figure does nothing. A gesture that
outlives the node it was editing, or that is caught by a background task, is
dropped.

**Reach** is nine panel pixels for the dot, a corner, the arrowhead or a circle,
and eight from an edge of the square or from the normal's segment. Both are
generous against the marks they draw, because the two misses do not cost the
same: a handle missed by two pixels orbits the scene, which the person then has
to undo by eye, while one caught a little early is released without motion and
does nothing.

**The order the reaches are tried is arrowhead, corner, dot, circle, edge,
normal segment**, with the nearest taken among the corners, among the circles
and among the edges. The dot and the circles sit inside the square they mark and
a corner is where two edges meet, so one nearest-thing search over all of them
at once would make the smaller handles unreachable. The normal's segment is last
because it leaves the centre, where every other handle already is, and a person
reaching for it has the whole of its length to reach for. The arrowhead is first
for the other half of that reason: it is a point at the far end of that same
segment, so anything tested before it would take the presses meant for it. One
consequence is worth knowing. Seen exactly down the normal the whole arrow
collapses onto the centre, so the head covers the dot and takes its presses;
that view is the aim's own best one, and a few degrees of lean pulls the head
clear and gives the dot back.

**Cursors.** The dot takes `Move` on hover and `Grabbing` while it is held. A
circle selects rather than moves, so it takes `PointingHand`, as every other
selectable mark in the window does. An edge and a corner both take the resize
cursor their orientation on screen names, and the direction is the difference
between them: an edge is dragged **across** itself, so its cursor is the
perpendicular of the edge, while a corner travels **along** the arc it turns on,
so its cursor is the perpendicular of its own radius from the centre, which is
that arc's tangent. Running the pointer down an edge and onto the corner turns
the cursor by exactly the difference between the two gestures. There is no
rotation cursor to give a corner instead; egui's set is the CSS one, and neither
has ever had such a thing. The normal's segment is an edge's case turned around,
being dragged along itself rather than across, so it is handed its own
perpendicular and answers with the cursor lying on the line. The arrowhead takes
the cursor of the gesture it is about to make: `AllScroll` for an aim, which is
free in two directions at once, and for a swing the corner's own reading, the
resize cursor along the arc the head travels. Windows draws `AllScroll` and
`Move` with the same glyph, both landing on `IDC_SIZEALL`, which is expected
rather than a mistake; the distinction is for the code's own clarity and for the
platforms that render the two apart. Image Detail's outline reads its edges and
corners the same way, so a corner under the pointer asks for the same cursor in
either panel.

**The pointer is a ray of the viewport's camera**, where in Image Detail it is a
ray of a reconstruction camera. Everything else about reading it is the same
idea: the ray is met with the patch's own geometry, and the edit is stated in
the patch's own terms. The ray is taken back through the node's similarity
first, because the figure is drawn in the world while the core steps act in the
reconstruction's own coordinates. A similarity preserves angles, so the
degenerate-view tests below read the same on either side of it.

**Which geometry the ray is met with is the handle's**, decided at the press and
carried for the whole drag, so both ends of one gesture are read the same way.
Both ends are places on that geometry rather than pixels of the window, because
what the gesture means is a statement about the patch and should not depend on
where the figure happened to be drawn.

| Handle | The ray is met with | What it asks for |
|---|---|---|
| Centre dot | the patch's plane | a translation by the travel since the press, read on `u` and `v` |
| Edge | the patch's plane | a resize to `(p + h) / 2`, `p` being the place's offset along that edge's own axis, with the far edge held |
| Corner | the patch's plane | a spin by the angle swept about `n`, both places read about `c` |
| Normal segment | the line `c + t n`, at its nearest point to the ray | a translation along `n` by the difference between the two places |
| Arrowhead | a plane through `c`, square to `n` or to the swing axis | a tilt to the normal the gesture names |
| Observation circle | nothing | the row selected in Track Edit, on a click |

The dot's travel has its component along `n` **set to zero by the handle**, and
that is where the constraint belongs. The two places are meetings of a ray with
the patch's plane, so they lie in it only to their last bits, and the dot is the
handle that means "across the plane and nowhere else". Reading the travel on `u`
and `v` alone says so outright, rather than leaving the step to project a
displacement it was never told was meant to be tangential. Both the dot and the
normal's segment read a **difference** and not a place, which is what keeps the
press's own offset from the centre: a dot grabbed a little off centre does not
jump under the pointer, and a segment grabbed at its tip does not fling the
patch out to where the tip was.

**The degenerate views are refused at the press.** Three handles name a point of
the patch's plane, and a plane seen edge-on turns a pixel of pointer motion into
an unbounded distance along it. When the angle between the view ray through `c`
and the plane is under 5 degrees, the dot, the edges and the corners take no
press and show no cursor, which is a refusal the person can see, the figure
being a line. The normal's segment is the mirror case: when the angle between
`n` and the view ray is under 5 degrees the nearest-point solve is
ill-conditioned and the segment takes no press either. That line is undirected,
so the angle is read as a magnitude; a view straight up the normal is exactly as
bad as one straight down it. A drag whose handle becomes refused while the
button is down is dropped.

The two refusals are complementary rather than merely alike, and the reason is
that they are one number read at two bars. Both are decided by the magnitude of
the cosine between the normal and the view ray through the centre: one refuses
the view where that cosine goes to zero, the other where it goes to one. The
nearest-point solve divides by `1 - (d . n)^2`, the square of the sine that the
plane's own reading is multiplied by. So the view in which one set of handles
dies is the view in which the other is at its best, and between them some handle
is always live.

### The arrowhead's two gestures

The arrowhead needs no refusal of its own, because its two gestures are each
other's cure. The aim reads a plane that is square to the view exactly when the
normal points along it; the swing reads an axis that is well determined exactly
when it does not. Both planes run through `c`, so neither can fall behind an eye
that is looking at the figure at all, and the arrowhead therefore answers from
every view it is drawn in.

Which gesture a press makes is decided by where the normal points. With `e` the
unit vector from `c` to the eye, the arrowhead **aims** when `|n . e|` is above
`cos 45 degrees` and **swings** when it is below. The magnitude and not the
signed value, because a patch showing its back is as square to the view as one
showing its face. That is the same cosine the two refusals read, at a third bar.
The press decides it, as the press decides the handle, so a gesture does not
change character halfway through, and the axis a swing turns about travels with
the drag rather than being read again each frame off an eye that has since
moved.

**Aiming**, when the normal lies near the line of sight. The pointer's ray is
met with the plane through `c` square to `n`, and the answer is the old normal
on a lever of `4h`, displaced by however far the pointer has travelled across
that plane since the press, then normalized. The plane and the lever are the
ones fixed at the press, so the gesture is a single map from the window onto the
sphere of normals rather than a thing that moves as it is used. It is the
**travel** and not the meeting itself that is read, for the reason the dot's
press offset is kept: a press that took the arrowhead is not standing where `c`
projects, the head being drawn out along the normal, and reading the meeting
outright would turn the normal over before the pointer had moved at all, which a
drag that ends where it started is not allowed to do.

Two properties follow from the lever. `4h` of travel is 45 degrees of tilt while
the arrowhead is drawn `2h` out, and the travel is read on the very plane the
arrow stands out of, so `4h` is twice the arrow's own drawn length whatever the
zoom. The aim is half as sensitive as the figure looks, at every distance, and a
small correction is a small motion, which is what the handle is for: a normal is
read off a surface a few degrees at a time. And because the travel is square to
`n`, the answer keeps the whole `4h` along it; the sum of a fixed vector and one
square to it can never turn through a right angle, so one gesture turns the
normal by less than 90 degrees and cannot push it through the square at all.

**The `4h` is a lever and not a distance the plane stands at**, and that
distinction is the whole of why the plane runs through `c`. Standing the plane
`4h` out along `n` gives the same arithmetic wherever the eye is far off and
fails where it is not. The standoff would be measured *toward* an eye the aim
was chosen for, so an eye within `4h` of a patch facing it has the plane behind
it, the ray meets nothing, and the press falls through to the viewport's
navigation with the arrowhead's cursor still showing. Close in on a patch that
faces you is not a corner case; it is the view a person zooms to when they mean
to work on a normal. Reading `c`'s own plane also takes the eye's distance out
of the gesture entirely. What a pixel of pointer travel is worth on a plane
depends on how far off that plane is, so a plane at a fixed standoff would make
the handle's sensitivity a function of the zoom, while `c`'s own plane and the
patch project alike and the ratio between them is the same at every zoom.

**Swinging**, when the normal lies across the line of sight. The normal turns
about one axis, which lies **in the patch's own plane**: the unit vector there
nearest the eye, which is the part of `e` square to `n`, normalized. The normal
therefore stays in the one plane through it square to that axis and never rolls
toward or away from the viewer, which is the motion that is hardest to aim and
easiest to overshoot when the arrowhead is nearly side-on. The pointer's ray is
met with the plane through `c` square to the axis, which is the plane the
arrowhead travels in and, the axis pointing at the eye as nearly as the square
allows, the plane most nearly facing the window. The angle swept about `c` from
the press's meeting to the pointer's is the angle the normal turns by. That is
the corner's own reading with the swing axis in place of the normal, which is
why the two take the same cursor.

A single reading serves the corner and the swing, taking a centre, an axis and
two places, and dropping each place's component along the axis first so that a
meeting sitting a rounding off the plane names the direction directly under it.
The angle is taken straight from the pair as `atan2` of the cross product's
component along the axis over the dot product, which already answers in
`(-pi, pi]`, rather than as a difference of two `atan2`s that would then have to
be brought back into range.

Both gestures state their answer as a normal, so `tilt_patch` knows nothing
about which one named it. The step turns the patch by the **least rotation**
carrying the old normal onto the new one, so a tilt adds no spin about the
normal; spin is the corners' job. A swing already is that rotation, its axis
being square to both normals by construction. The step also stops the turn 80
degrees short of any observation's camera, so dragging the arrowhead against the
limit traces the edge of what the existing observations can see, and the
version's sentence says which observation stopped it. That rule, and what each
edit does to the sightings, are
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Placing,
sizing and turning by hand"; the drag and the wire call reach it through the
same step, so they cannot stop in different places.

### The handles at infinity

For a direction patch the pointer is its ray's **direction** alone, met with the
tangent plane the way an observation's is: the viewer is in effect at the
sphere's centre, so the ray's origin drops out. The dot, the edges and the
corners work from that place exactly as they do on a finite patch, through the
same core steps, which already carry a `w = 0` patch: a translation moves the
bearing and renormalizes it, a resize changes the tangent half-length, and a
spin turns the square about the bearing. The normal's segment and the arrowhead
do not exist, and what they would ask for is refused in core's own sentence: a
translation with a component along the normal, and a tilt, each answer a `w = 0`
track with `TrackEditError::AtInfinity`, which the wire tools pass on.

The tangent plane is never edge-on, so the plane refusal cannot fire and the
normal's is always on. The one refusal a track at infinity has is a pointer more
than 85 degrees from the bearing, where the tangent point runs off to infinity.

## Parameters

Every value here is a constant of the module named beside it, not a setting.

| Parameter | Value | Meaning |
|---|---|---|
| `CIRCLE_SEGMENTS` | `24` | segments in the centre disc and in each mark's circle (`viewer_3d/bench_track.rs`) |
| `CIRCLE_RADIUS` | `1/8` | the disc's and the circles' radius, in half-lengths |
| `PLANE_LIFT` | `1e-3` | how far in-plane geometry is lifted along the normal, in half-lengths |
| `NORMAL_LENGTH` | `2.0` | how far the normal's segment stands off the square, in half-lengths |
| `BARB_BACK` | `0.25` | how far back along the normal the barbs reach, as a fraction of its length |
| `BARB_SIDE` | `0.1` | how far to either side they reach, in the same units |
| `FOG_DISTANCE` | `4.0` | how far behind the scene the figure fades to the floor, in half-lengths |
| `SELECTED_CIRCLE_SCALE` | `1.6` | how much larger the selected observation's circle is drawn |
| `HANDLE_HIT_RADIUS` | `9.0` | reach of the dot, a corner, the arrowhead and a circle, in panel px |
| `EDGE_HIT_WIDTH` | `8.0` | reach of an edge and of the normal's segment, in panel px |
| `BENCH_LINE_HALF_WIDTH` | `1.25` | half the stroke width, in px (`scene_renderer/gpu_types.rs`) |
| `BEHIND_OPACITY` | `0.45` | opacity the moment the figure passes behind the scene (`shaders/bench_track.wgsl`) |
| `FLOOR_OPACITY` | `0.15` | the opacity it fades to and never below |
| `INF_DEPTH` | `1e-6` | the NDC depth a direction's fragments are pinned at |
| `MIN_PLANE_ANGLE_DEG` | `5.0` | the edge-on and end-on refusals, in degrees (`bench/geometry.rs`) |
| `MAX_BEARING_ANGLE_DEG` | `85.0` | how far from a bearing a pointer may point, in degrees |
| `AIM_ANGLE_DEG` | `45.0` | how near the line of sight the normal must lie for the arrowhead to aim |
| `AIM_LEVER` | `2 * NORMAL_LENGTH` | the aim's lever, in half-lengths, stated in terms of the arrow's own length |
| `MAX_TILT_DEG` | `80.0` | how far from any observation a tilt stops, in degrees (`sfmtool-core`) |

`AIM_LEVER` is written in terms of `NORMAL_LENGTH` rather than as a second
literal, because the arrow a person sees and the travel their pointer turns it
by are two facts about one handle and a second literal could drift from the
first silently.

## Testing

`viewer_3d/bench_track/tests.rs` asserts the figure without a window or a
device, against a demo reconstruction rewritten so that every keypoint is its
point's exact projection: the instance list holds the square, the disc, the
arrow and one mark per observation with a keypoint; each mark's `q_i` reprojects
onto the very keypoint it was unprojected from; a track at infinity holds its
square, disc and marks with `w = 0` endpoints and no normal; a cluster-stage
item and a track with no patch draw nothing; the marks carry their verdicts
while the square and the normal stay in the `in` colour; and the fog distance is
four half-lengths.

`scene_renderer/bench_track/tests.rs` runs the upload and the pipelines against
wgpu's `noop` backend, where wgpu-core still performs its full validation while
wgpu-hal stubs the driver: a vertex layout that disagrees with the instance
struct, or a uniform block whose Rust side has drifted from the WGSL, is a
validation error there rather than a blank pass on a real GPU. The depth-aware
blend itself is checked by eye, with MCP screenshots of the viewport showing a
square lying on a surface, one in front of it and one behind it
([`mcp-server.md`](mcp-server.md)).

`bench/geometry/tests.rs` covers the ray arithmetic on its own: a ray meets a
finite patch where its plane is and a bearing's tangent plane at its own
direction over the bearing; a plane seen edge-on is refused and a bearing's
never is; the normal is refused end-on exactly where the plane is at its best; a
ray names the point of the normal's line it comes nearest; a turn on the plane
is the angle swept about the centre; the arrowhead aims within 45 degrees of the
view and swings outside it; an aim turns 45 degrees in four half-lengths and
never reaches 90; an aim answers from an eye closer in than its own lever, where
a plane stood off by that lever would sit behind the camera; and a swing turns
about the axis of the patch's plane nearest the eye.

`viewer_3d/tests.rs` drives whole headless frames through the viewport, as the
Image Detail handles' tests drive real frames: a press on an edge resizes and
orbits nothing while the same motion off it orbits; a dragged edge lands under
the release point with the far edge held; a corner dragged onto its neighbour is
a quarter turn and one version; a dot drag is one version whose label names the
move; Escape leaves no edit and a drag that ends where it started pushes
nothing; an edge-on view takes no press and a busy node takes none either; a
click on an observation's circle selects its row; the three plane handles of a
track at infinity move the bearing, the tangent size and the spin; a corner
turns the cursor along its arc where an edge lies across itself; a normal drag
moves the patch along its normal and orbits nothing; a view down the normal
refuses the segment where the plane handles are at their best; an arrowhead drag
tilts the patch and orbits nothing; an arrowhead close to the eye takes the
press rather than orbiting; and the arrowhead's gesture is chosen at the press
and does not change under it.

## Non-goals

- **Drawing the tracks that are not active.** The bench holds several items and
  this layer draws the one being worked on, as the Image Detail bench layer
  does. Clicking a track in the world to make it active is not something the
  layer offers; a track is activated from the Scene tree or Track Edit
  ([`track-edit.md`](track-edit.md)).
- **A pick-buffer entry.** Hit-testing is on the CPU, against the figure the
  pass drew projected through the same camera, so occlusion does not enter into
  it: a handle drawn through the point cloud is grabbed like any other. The
  figure's depth-aware blend is about what is seen, and the hit test about where
  the pointer is.
- **A gesture of its own for the cluster stage.** A cluster-stage item has one
  affine shape per sighting and no shared geometry, so its gestures are each
  sighting's own and live in the photograph
  ([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
  layer").
- **Anything for camera view mode.** The 3D view is always an undistorted
  perspective projection, and it is the photograph behind it that is warped,
  onto a distorted grid mesh. The layer is scene geometry and lines up as the
  point cloud does.
