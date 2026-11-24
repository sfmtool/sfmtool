# Overlay ribbons in the 3D view

**Status:** Draft

The 3D view draws several kinds of overlay figure into the scene. Today they are:
- the focused bench item's figure;
- the selected and shown measurements;
- the yellow-orange track rays from each camera to the selected point.

In time there will be more: GPS markers, scale bars, alignment pairs and other
widgets. Each is a set of camera-facing ribbons tested against the scene's depth.
This draft makes them one **overlay ribbon layer**, made of three parts:
- one instance format, in which a segment, a ring or a disc is given in world
  space, or a direction for a ray that goes to infinity, with its size in screen
  points;
- one buffer that holds every overlay's instances;
- producers that rebuild and upload their instances only when what they draw from
  changes, not every frame.

A new widget is then a producer. It needs no buffer, upload function or render
pass of its own. A track ray to a point at infinity reaches the horizon instead
of stopping at a fixed length.

Amends:
- [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md): the pass and
  instance format the bench figure and the measurements share today.
- [gui/measurement-view.md](../gui/measurement-view.md) § "Drawing measurements".
- [gui/architecture.md](../gui/architecture.md): Pass 4, the track rays, and
  Pass 5.
- [gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md): the pass list.

Decided by the maintainer:
- **The bench figure is rebuilt when the camera moves.** Its arrowhead's barbs
  turn to face the eye, so it is rebuilt then, as now. It is a few hundred
  instances. Whether to make the barbs turn in the shader instead is to be judged
  once the rest is built.
- **Edges are anti-aliased over about one panel point,** to match the egui
  feathering the measurements had before they moved into the scene.
- **Halos are drawn before strokes.** Within a producer, every halo is drawn
  before every stroke, so one measurement's halo never crosses another's stroke.
- **The track rays are included,** and a ray to a point at infinity is drawn to
  the horizon (§ "Track rays").
- **Track rays are drawn as the other overlays are:** anti-aliased strokes
  rather than the soft glow, and dimmed to `BEHIND_OPACITY` where the scene
  hides them rather than discarded, so a ray going into the bull shows where it
  goes.

Nothing in this draft is still open.

## What exists

**The bench figure and the measurements.** They share one shader
(`bench_track.wgsl`) and one instance format, `BenchEdgeInstance`: two
homogeneous world endpoints, a colour, a half width in physical pixels and a
`fade` flag. They are drawn in two passes. The measurements draw into their own
layer texture, which egui paints over the scene, and the figure draws into the
scene texture.

Both are rebuilt on the CPU and re-uploaded every frame, each into a new buffer
made with `create_buffer_init` (`SceneRenderer::upload_measurements`,
`upload_bench_track`). Their instances depend on the camera:
- **A measurement mark.** Its ring is 24 chords placed in screen space and
  unprojected at the mark's depth. Its dot is a short ribbon. Every piece is
  pulled toward the eye by a fraction that follows the field of view and the
  panel's height (`viewer_3d::measurements::ribbons`). That is about 100
  instances per measurement.
- **The figure's arrowhead.** Its barbs turn to face the eye.

With a few measurements none of this costs anything. With *Show all measurements*
on and a thousand measurements, it is about 100,000 instances (5.6 MB) built,
allocated and uploaded every frame while the camera orbits and nothing else
changes. The chord rings also overlap their own halo where chords meet, and the
ribbons are hard-edged.

**The track rays** have their own pass (`track_ray.wgsl`), their own instance
(`EdgeInstance`: two `vec3` endpoints) and a uniform half width of 1.5 pixels.
- **Look and occlusion.** Each ray is an orange glow: 40% opacity at its centre,
  falling to nothing at its edges. It is discarded wherever scene geometry is in
  front of it.
- **Rebuilding.** The rays are rebuilt only when the selected point, the version
  that holds it, or its node's transform changes (`app::selected_point_source`).
- **To a finite point,** a ray runs from the camera centre along the observation
  direction to the nearest point on that ray to the 3D point, so the gap shows
  the reprojection error.
- **To a point at infinity,** a ray runs along the observation direction for a
  fixed length, twice the diagonal of the camera centres' bounding box
  (`INFINITY_RAY_SCENE_MULTIPLE`, `camera_cloud_extent`), and stops in mid-air.

**Built once.** The target indicator and the frusta are built once and have their
own shaders; § "Non-goals".

## The design

### One instance format, independent of the camera

```rust
/// One piece of an overlay figure.
#[repr(C)]
pub(crate) struct OverlayInstance {
    /// A segment's two ends; a ring's or a disc's centre in `a`.
    /// Homogeneous world coordinates: `w = 1` a place, `w = 0` a direction,
    /// which is drawn at infinity.
    pub(crate) a: [f32; 4],
    pub(crate) b: [f32; 4],
    /// Not premultiplied, in the colour space of the target it is drawn into.
    pub(crate) color: [f32; 4],
    /// Half the stroke's width, in panel points.
    pub(crate) half_width: f32,
    /// A ring's or a disc's radius, in panel points; 0 for a segment.
    pub(crate) radius: f32,
    /// Segment, ring or disc.
    pub(crate) kind: u32,
    /// How it is drawn behind the scene (FADE or HOLD), and
    /// PULL_TOWARD_EYE.
    pub(crate) flags: u32,
}
```

The vertex shader already has the camera's uniform block, and everything that
depends on the camera moves into it.

**Sizes in points.** Widths and radii are in panel points. The shader turns them
into pixels with a new `pixels_per_point` uniform, so an instance stays the same
across a DPI change.

**The pull toward the eye.** A flagged piece is moved toward the eye by the
fraction `PULL_POINTS · 2·tan(vfov/2) / panel_height`, capped as today. The CPU
computes it once per frame and hands it to the shader as a uniform. A piece moves
along its own line of sight, so where it projects is unchanged and the CPU hit
test agrees with it.

**Rings and discs.** Each is one instance. The shader projects the centre and
expands a quad of `radius + half_width` points, aligned to the screen, around it.
The fragment shader draws the annulus or disc from each fragment's distance to
the centre. That replaces 24 chord instances, a halo and a dot with three
instances (halo ring, ring, dot), and it removes the doubled halo where chords
met.

**Segments** are today's ribbons, with two changes:
- **Clipping.** They are clipped at the near plane in homogeneous view space, so
  an end with `w = 0` stays a direction rather than being divided by zero
  (§ "Track rays").
- **Edges.** A segment's edges are anti-aliased over about one panel point.

**Behind the scene.** Each instance says how its hidden part is drawn:
- `FADE`: fades with depth behind the scene, down to a floor. This is the bench
  figure.
- `HOLD`: held at `BEHIND_OPACITY`. This is a measurement and a track ray, and
  also how a highlighted surfel's hidden outline is drawn.

**Unchanged.** The bench figure's arrowhead barbs still turn to face the eye, so
the figure's producer rebuilds when the camera moves (§ "Producers and their
keys").

### One buffer, a region per producer

The scene renderer owns one `OverlayRibbons` store:

```rust
pub(crate) struct OverlayRibbons {
    /// Grown by doubling when a producer outgrows its region, never shrunk
    /// within a session.
    buffer: wgpu::Buffer,
    /// In draw order.
    regions: Vec<Region>,
}

struct Region {
    producer: ProducerId,
    target: OverlayTarget,
    /// Where its instances start in `buffer`, how many it holds now, and how
    /// many it has room for.
    offset: u64,
    count: u32,
    capacity: u32,
}

pub(crate) enum OverlayTarget {
    /// The scene texture, blended in linear light (the bench figure, the
    /// track rays).
    Scene,
    /// The measurements' layer, blended on encoded values and painted over the
    /// scene by egui (§ "Drawing measurements").
    MeasurementLayer,
}
```

- **Writing.** A producer whose instances changed hands them to
  `OverlayRibbons::write(producer, target, &[OverlayInstance])`.
  - If they fit its region's capacity, that is one `queue.write_buffer` of that
    region and nothing else.
  - If they don't, the region's capacity doubles, the regions are repacked, and
    the store rewrites the regions that moved.
  - A producer that changed nothing writes nothing, and its region keeps the last
    frame's instances.
- **Drawing.** Each target's pass binds the buffer once and draws each of its
  regions with `draw(0..6, offset..offset + count)`, in region order. Each target
  has one pipeline: the two that exist today, with the new vertex layout.
- **Order.** A producer lays out its own instances with all halos first, then all
  strokes, then dots, so one item's halo never crosses another's stroke. The
  regions are drawn in a fixed order: track rays, then the bench figure on the
  scene target; measurements on their layer.
- **Retiring.** A producer that draws nothing writes an empty slice and its
  region's count drops to 0. Its capacity is kept for next time.

The store is the one place overlay geometry lives on the GPU. The per-overlay
buffers go away: `measurement_edge_buffer`, `bench_track_edge_buffer` and
`track_ray_edge_buffer`, with their counts. The track rays' own pass and pipeline
go away too.

### Producers and their keys

A producer is the code that turns one overlay's data into instances. It keeps the
key it last built from, and rebuilds only when that key changes:

| Producer | Target | Key |
|---|---|---|
| Track rays | `Scene` | The selected point and the version that holds it (`app::selected_point_source`), the node's transform and visibility, and whether the observations' keypoints are cached |
| Bench figure | `Scene` | The focused item's `Arc` and the camera's pose, for the barbs |
| Measurements | `MeasurementLayer` | Each drawn node's document serial and display transform, the selected measurement and endpoint, the focused item's `Arc`, *Show all measurements*, and the visible nodes |

The measurement key covers everything the instances are built from, now that
nothing in them depends on the camera. It reuses the readings cache's key
(`SceneNode::measurement_readings`, `SceneNode::item_reading`) with the selection
and display state added. Orbiting, zooming and resizing the panel then rebuild
nothing and upload nothing. The track rays already rebuild only on their key, so
they keep that behaviour.

The hit test (`measurement_marks`) keeps projecting the endpoints on the CPU from
the same readings. It does not read the instances.

A new widget, such as a GPS marker, is another producer. It builds
`OverlayInstance`s from its own data, chooses a target and a key, and calls
`write`. Its labels and hover text stay egui, as the measurements' do.

### Track rays

A track ray is a segment in the rays' orange, anti-aliased as the other
overlays are, with `HOLD` behind the scene: where geometry hides it, it is drawn
at `BEHIND_OPACITY`, so a ray going into the bull shows where it goes. Its width
is a stroke in points like a measurement's, replacing the 1.5-pixel glow. It runs from the observing camera's centre (`a`,
`w = 1`) to its far end (`b`):

- **To a finite point,** `b` is the nearest point on the observation ray to the
  3D point, with `w = 1`, as today.
- **To a point at infinity,** `b` is the observation direction in world space
  with `w = 0`. The fixed length, `INFINITY_RAY_SCENE_MULTIPLE` and
  `camera_cloud_extent` go away. In the shader:
  - **Where it ends on screen.** The direction projects to its vanishing point,
    so the ray runs from the camera's place to where that direction meets the
    horizon. A point at infinity is drawn at that same spot
    ([point-cloud-rendering.md](../gui/point-cloud-rendering.md) § "Points at
    Infinity"), so every ray of its track ends on it, the way rays to a finite
    point end at or near it.
  - **Its depth.** The direction's end is pinned at the same tiny positive NDC
    depth the points and patches at infinity use. That puts it behind every
    finite thing, so a ray is dimmed where finite geometry covers it.
  - **Behind the camera.** If the direction points away from the camera, the
    segment is clipped at the near plane in homogeneous coordinates. The part in
    front of the camera, from the camera centre outward, is drawn up to the
    screen's edge, where the clipped line leaves the view.
- **Through the node's transform.** The node's display transform is applied to a
  direction as a rotation only: a similarity's translation and scale do not move
  a direction. This matches how points at infinity are drawn.

That is the whole change for points at infinity. A ray now reaches its point on
the screen wherever the point is drawn, instead of ending at a length that looks
arbitrary next to it.

## Non-goals

- **The target indicator and the frusta** are built once, and have their own
  shaders: the indicator's glow and the frusta's per-node bundles.
- **The hidden part of a highlighted outline** is drawn on the surfels' own
  geometry, not as ribbons.
- **Labels and hover text** stay egui.

## Testing

**The store**
- A producer that writes within its capacity issues one write of its own region,
  and leaves the other regions untouched.
- Growing past capacity repacks, and every region keeps its instances.
- A key that hasn't changed writes nothing.
- An empty write draws nothing and keeps the region's capacity.
- Regions are drawn in the fixed order, and halos before strokes within a
  producer.

**The measurement producer**
- It builds once, and doesn't rebuild when only the camera, the field of view or
  the panel's size changes.
- It does rebuild on each change to its key: a step on the focused item, an undo,
  another measurement selected, *Show all* toggled, a node hidden, the display
  transform changed.

**The track-ray producer**
- A finite point's rays end where they do today.
- A point at infinity's rays carry `w = 0` and the direction through the node's
  rotation only. A node transform with scale and translation leaves the
  direction unchanged except for the rotation.
- There is no fixed-length end.
- In the shader-side geometry, a `w = 0` end projects to the same screen spot as
  the point at infinity's own splat.

**Instances**
- A ring or disc is one instance, with its radius in points.
- The flags: `HOLD` and `PULL_TOWARD_EYE` on a measurement's pieces, `FADE` on
  the figure's, `HOLD` and no pull on a ray's.

**Pass contract.** Each target's pipeline draws regions from the shared buffer on
the `noop` device, with the new vertex layout.

**By eye, on a real GPU, in the PR's screenshots**
- Rings anti-aliased, with an even halo.
- Hidden measurement pieces at `BEHIND_OPACITY`.
- No visible change to the bench figure.
- The hit test still landing on the drawn mark at several fields of view.
- For a point at infinity, every ray of its track reaching its splat at the
  horizon: seoul_bull v02 has one, point #264.
- A track ray that passes into the bull drawn at `BEHIND_OPACITY` where the bull
  hides it, and at full strength where it is in front.

**Cost.** With 1,000 measurements and *Show all* on, a frame while orbiting
builds and uploads no instances. A test counter checks this, as for
`item_reading`.

## Steps

1. **The store.** `OverlayRibbons` with regions, `write`, per-target draws, and
   the fixed region order. The measurements and the bench figure move onto it,
   still built every frame, with today's instance contents in the new format.
   There is no visible change.
2. **Instances independent of the camera.** This step brings:
   - widths and radii in points;
   - the pull toward the eye in the shader;
   - rings and discs as one instance each;
   - edges anti-aliased over a point;
   - halos drawn before strokes;
   - the measurement producer keyed and cached.

   It is visible (rings, edges and halo order), so it needs screenshots.
3. **Track rays.** The rays move onto the store as a producer, drawn as
   anti-aliased strokes and dimmed where hidden, and their own pass is removed.
   A ray to a point at infinity becomes a direction drawn to the horizon. This is
   visible, so it needs screenshots.
4. **Later, once the rest is built:** judge whether the bench figure's barbs
   should turn in the shader, so the figure's key no longer includes the camera.
