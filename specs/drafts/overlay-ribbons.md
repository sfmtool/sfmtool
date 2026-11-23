# Overlay ribbons in the 3D view

**Status:** Draft

The 3D view draws several kinds of overlay figure into the scene: the focused
bench item's figure, the selected and shown measurements, and in time GPS
markers, scale bars, alignment pairs and other widgets. Each is a set of
camera-facing ribbons tested against the scene's depth. This draft makes them
one **overlay ribbon layer**: one instance format that describes a segment, a
ring or a disc in world space with its size in screen points; one buffer that
holds every overlay's instances; and producers that rebuild and upload their
instances only when what they draw from changes, not on every frame. A new
widget becomes a producer, with no new buffer, upload function or render pass of
its own.

Amends [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) (the
pass and instance format the bench figure and the measurements share today),
[gui/measurement-view.md](../gui/measurement-view.md) § "Drawing measurements",
[gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md) (the pass list)
and [gui/architecture.md](../gui/architecture.md) (Pass 5).

Open: § "Open questions".

## What exists

The bench figure and the measurements share one shader (`bench_track.wgsl`) and
one instance format, `BenchEdgeInstance`: two homogeneous world endpoints, a
colour, a half width in physical pixels and a `fade` flag. They are drawn in two
passes. The measurements draw into their own layer texture, which egui paints
over the scene; the figure draws into the scene texture.

Both are rebuilt on the CPU and re-uploaded on every frame, each into a new
buffer made with `create_buffer_init` (`SceneRenderer::upload_measurements`,
`upload_bench_track`). They must be rebuilt every frame because their instances
depend on the camera:

- **Measurements.** A mark's ring is 24 chords placed in screen space and
  unprojected at the mark's depth, its dot is a short ribbon, and every piece is
  pulled toward the eye by a fraction that follows the field of view and the
  panel's height (`viewer_3d::measurements::ribbons`). That is about 100
  instances per measurement.
- **The bench figure.** The arrowhead's barbs turn to face the eye. It is a few
  hundred instances.

That costs nothing with a few measurements. With *Show all measurements* on and
a thousand measurements, it is about 100,000 instances (5.6 MB) built, allocated
and uploaded on every frame, while the camera orbits and nothing else changes.
The chord rings also overlap their own halo where chords meet, and the ribbons
are hard-edged where the egui overlay they replaced was anti-aliased.

The track rays (their own `EdgeInstance` and pass) upload only when the selected
point changes. The target indicator and the frusta are geometry that is built
once. Neither is a ribbon of this kind; § "Non-goals".

## The design

### One instance format, independent of the camera

```rust
/// One piece of an overlay figure.
#[repr(C)]
pub(crate) struct OverlayInstance {
    /// A segment's two endpoints; a ring's or a disc's centre in `a`.
    /// Homogeneous world coordinates: `w = 1` a place, `w = 0` a direction.
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
    /// FADE_BEHIND: the part behind the scene fades with depth (the bench
    /// figure) instead of holding at BEHIND_OPACITY (a measurement).
    /// PULL_TOWARD_EYE: drawn PULL_POINTS toward the eye (a measurement's
    /// pieces).
    pub(crate) flags: u32,
}
```

Everything that depends on the camera moves into the vertex shader, which already
has the camera's uniform block:

- **Sizes in points.** Widths and radii are in panel points. The shader turns
  them into pixels with `pixels_per_point`, a new uniform, so an instance stays
  the same across a DPI change.
- **The pull toward the eye.** The shader moves a flagged piece's world position
  toward the eye by the fraction `PULL_POINTS · 2·tan(vfov/2) / panel_height`,
  capped as today. The CPU computes that fraction once per frame from the same
  numbers and puts it in a uniform. A piece moves along its own line of sight,
  so where it projects is unchanged and the CPU hit test agrees with it.
- **Rings and discs.** A ring or disc is one instance. The shader projects its
  centre and expands a screen-aligned quad of `radius + half_width` points
  around it. The fragment shader draws the annulus or disc from the distance to
  the centre, anti-aliased over about one pixel. That replaces 24 chord
  instances, a halo and a dot with three instances (halo ring, ring, dot). It
  also removes the doubled halo where chords met.
- **Segments** are today's ribbons, anti-aliased across their width in the same
  way. Near-plane clipping in the vertex shader is kept.

The bench figure's arrowhead barbs still turn to face the eye, so the figure's
producer rebuilds them when the camera moves (§ "Producers and their keys"). A
barb could instead become its own instance kind whose direction the shader
computes; § "Open questions".

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
    /// The scene texture, blended in linear light (the bench figure).
    Scene,
    /// The measurements' layer, blended on encoded values and painted over the
    /// scene by egui (§ "Drawing measurements").
    MeasurementLayer,
}
```

- **Writing.** A producer whose instances changed hands them to
  `OverlayRibbons::write(producer, target, &[OverlayInstance])`. If they fit its
  region's capacity, that is one `queue.write_buffer` of that region and
  nothing else. If they don't, the region's capacity doubles, regions are
  repacked, and the store rewrites the regions that moved. A producer that
  changed nothing writes nothing, and its region keeps the last frame's
  instances.
- **Drawing.** Each target's pass binds the buffer once and draws each of its
  regions with `draw(0..6, offset..offset + count)`, in region order. There is
  one pipeline per target: the two that exist today, with the new vertex layout.
- **Retiring.** A producer that draws nothing writes an empty slice, and its
  region's count drops to 0. Its capacity is kept for the next time.

This makes the store the one place where overlay geometry lives on the GPU. The
per-overlay fields (`measurement_edge_buffer`, `bench_track_edge_buffer` and
their counts) go away.

### Producers and their keys

A producer is the code that turns one overlay's data into instances. It keeps
the key it last built from, and rebuilds only when the key changes:

| Producer | Target | Key |
|---|---|---|
| Measurements | `MeasurementLayer` | Each drawn node's document serial and display transform, the selected measurement and endpoint, the focused item's `Arc`, *Show all measurements*, and the visible nodes |
| Bench figure | `Scene` | The focused item's `Arc` and the camera's pose, for the barbs (§ "Open questions") |

The measurement key holds everything the instances are built from, now that
nothing in them depends on the camera. It uses the readings cache's key
(`SceneNode::measurement_readings`, `SceneNode::item_reading`) together with the
selection and display state. Orbiting, zooming and resizing the panel then
rebuild nothing and upload nothing.

The hit test (`measurement_marks`) keeps projecting the endpoints on the CPU from
the same readings. It does not read the instances.

A new widget, such as a GPS marker, is a producer: it builds `OverlayInstance`s
from its own data, chooses a target and a key, and calls `write`. Its labels and
hover text stay egui, as the measurements' do.

## Non-goals

- **The track rays** have their own pipeline, width and depth behaviour, and
  already upload only on change. They could become a producer later. Moving them
  is not part of this draft.
- **The target indicator and the frusta** are built once and have their own
  shaders (the indicator's glow, the frusta's per-node bundles).
- **The hidden part of a highlighted outline** is drawn on the surfels' own
  geometry, not as ribbons.
- **Labels and hover text** stay egui.

## Open questions

1. **The arrowhead's barbs.** Keep rebuilding the figure whenever the camera
   moves (a few hundred instances), or add a `Barb` kind whose direction the
   shader turns toward the eye, so the figure's key no longer holds the camera?
   The first is simpler, and its cost is small next to what this draft saves.
2. **The anti-aliasing width.** One pixel at the panel's `pixels_per_point`, or
   about a point, to match egui's feathering? The second matches how the
   measurements looked before the overlay moved into the scene.
3. **Ring and line drawing order inside a measurement.** Today the halos are
   drawn under each stroke, piece by piece. With one buffer region per producer,
   should all halos be drawn before all strokes, so that a stroke is never
   crossed by another measurement's halo? That changes how crossing
   measurements look.

## Testing

- **Store.** A producer writing within its capacity issues one write of its own
  region, and leaves the other regions untouched. Growing past capacity repacks
  and keeps every region's instances. An unchanged key writes nothing. An empty
  write draws nothing and keeps the capacity.
- **Measurement producer.** It builds once and does not rebuild when only the
  camera, the field of view or the panel size changes. It rebuilds on each key
  change: a step on the focused item, an undo, selecting another measurement,
  toggling *Show all*, hiding a node, and changing the display transform.
- **Instances.** A ring or disc is one instance with its radius in points. The
  pull flag is set on a measurement's pieces and not on the figure's.
  `FADE_BEHIND` is set on the figure's pieces and not on a measurement's.
- **Pass contract.** Each target's pipeline draws regions from the shared buffer
  on the `noop` device, with the new vertex layout.
- **By eye, on a real GPU,** in the PR's screenshots: rings anti-aliased with an
  even halo; hidden pieces at `BEHIND_OPACITY`; no visible change for the bench
  figure; the hit test still landing on the drawn mark at several fields of
  view.
- **Cost.** With 1,000 measurements and *Show all* on, a frame while orbiting
  builds and uploads no instances. That is checked by a test counter, as for
  `item_reading`.

## Steps

1. **The store.** `OverlayRibbons` with regions, `write` and per-target draws.
   The measurements and the bench figure move onto it, still built every frame,
   with today's instance contents in the new format. No visible change.
2. **Instances independent of the camera.** Widths and radii in points, the pull
   in the shader, rings and discs as one anti-aliased instance each, and the
   measurement producer keyed and cached. This is visible (rings and edges), so
   it needs screenshots.
3. **The bench figure's key**, as § "Open questions" 1 decides.
