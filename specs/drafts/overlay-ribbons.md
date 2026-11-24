# Overlay ribbons in the 3D view

**Status:** Draft. Steps 1 and 2, the store and instances independent of the
camera, are built and specified in
[gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The overlay
ribbon store" and [gui/measurement-view.md](../gui/measurement-view.md) §
"Drawing measurements"; steps 3 and 4 are not built.

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
- [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The overlay
  ribbon store": the track rays as a third producer.
- [gui/architecture.md](../gui/architecture.md): Pass 4, the track rays, and
  Pass 5.
- [gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md): the pass list.
- [gui/scene-graph.md](../gui/scene-graph.md) § "Track rays and background
  image".

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

**The store and the instance format.** The bench figure and the measurements
are drawn out of one buffer, a region per producer, and each piece is one
`OverlayInstance`: a segment, a ring or a disc, with homogeneous world places,
sizes in panel points, flags for how it is drawn behind the scene and
whether it is pulled toward the eye, and a segment's inset at each end, by which
a measurement's line stops at its rings on screen. Segments are clipped at the
near plane in homogeneous view space, so an end with `w = 0` stays a direction,
and edges are anti-aliased over about a point
([gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The overlay
ribbon store"). The measurements are built only when their key changes; the
bench figure every frame, for its barbs.

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

### The track rays as a producer

The track rays become the store's third producer, drawn first on the `Scene`
target, before the bench figure. Their own buffer, `track_ray_edge_buffer` with
its count, their pass and their pipeline go away.

| Producer | Target | Key |
|---|---|---|
| Track rays | `Scene` | The selected point and the version that holds it (`app::selected_point_source`), the node's transform and visibility, and whether the observations' keypoints are cached |

The track rays already rebuild only on their key, so they keep that behaviour.

### Track rays

A track ray is a segment in the rays' orange, anti-aliased as the other
overlays are, with `HOLD` behind the scene and no pull toward the eye: where
geometry hides it, it is drawn at `BEHIND_OPACITY`, so a ray going into the bull
shows where it goes. Its width is a stroke in points like a measurement's,
replacing the 1.5-pixel glow. It runs from the observing camera's centre (`a`,
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
    segment is clipped at the near plane in homogeneous coordinates, as every
    segment already is. The part in front of the camera, from the camera centre
    outward, is drawn up to the screen's edge, where the clipped line leaves the
    view.
  - **Occlusion of a mixed segment.** A segment with one place and one
    direction is neither wholly finite nor wholly at infinity, so the shader's
    choice of fading or holding behind the scene, which today follows whether
    both ends are directions, has to hold for every fragment of it; `HOLD` does.
- **Through the node's transform.** The node's display transform is applied to a
  direction as a rotation only: a similarity's translation and scale do not move
  a direction. This matches how points at infinity are drawn.

That is the whole change for points at infinity. A ray now reaches its point on
the screen wherever the point is drawn, instead of ending at a length that looks
arbitrary next to it.

### The bench figure's barbs

The bench figure's producer rebuilds every frame because its arrowhead's barbs
turn to face the eye. Once the rest is built, judge whether the barbs should
turn in the shader instead, so that the figure's key no longer holds the
camera's pose and a camera move writes nothing for it either.

## Non-goals

- **The target indicator and the frusta** are built once, and have their own
  shaders: the indicator's glow and the frusta's per-node bundles.
- **The hidden part of a highlighted outline** is drawn on the surfels' own
  geometry, not as ribbons.
- **Labels and hover text** stay egui.

## Testing

**The track-ray producer**
- A finite point's rays end where they do today.
- A point at infinity's rays carry `w = 0` and the direction through the node's
  rotation only. A node transform with scale and translation leaves the
  direction unchanged except for the rotation.
- There is no fixed-length end.
- In the shader-side geometry, a `w = 0` end projects to the same screen spot as
  the point at infinity's own splat.
- The flags: `HOLD` and no pull on a ray's pieces.
- A key that hasn't changed writes nothing.
- Regions are drawn in the fixed order: the track rays, then the bench figure.

**Pass contract.** The scene target's pipeline draws the rays' region from the
shared buffer on the `noop` device, and the rays' own pass is gone.

**By eye, on a real GPU, in the PR's screenshots**
- For a point at infinity, every ray of its track reaching its splat at the
  horizon: seoul_bull v02 has one, point #264.
- A track ray that passes into the bull drawn at `BEHIND_OPACITY` where the bull
  hides it, and at full strength where it is in front.

## Steps

1. **The store.** *Built*
   ([gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The
   overlay ribbon store").
2. **Instances independent of the camera.** *Built*: widths and radii in
   points, the pull toward the eye in the shader, rings and discs as one
   instance each, edges anti-aliased over a point, halos before strokes, and
   the measurement producer keyed and cached
   ([gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The
   overlay ribbon store", [gui/measurement-view.md](../gui/measurement-view.md)
   § "Drawing measurements").
3. **Track rays.** The rays move onto the store as a producer, drawn as
   anti-aliased strokes and dimmed where hidden, and their own pass is removed.
   A ray to a point at infinity becomes a direction drawn to the horizon. This is
   visible, so it needs screenshots.
4. **Later, once the rest is built:** judge whether the bench figure's barbs
   should turn in the shader, so the figure's key no longer includes the camera
   (§ "The bench figure's barbs").
