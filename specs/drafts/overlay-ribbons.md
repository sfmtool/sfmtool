# Overlay ribbons in the 3D view

**Status:** Draft. Steps 1 to 3, the overlay ribbon store, instances
independent of the camera and the track rays as its third producer, are built
and specified in [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md)
§ "The overlay ribbon store", [gui/measurement-view.md](../gui/measurement-view.md)
§ "Drawing measurements" and [gui/architecture.md](../gui/architecture.md) §
"Track Ray Visualization". Step 4 is not built.

The 3D view draws its overlay figures, the selected point's track rays, the
focused bench item's figure and the drawn measurements, out of one buffer, the
**overlay ribbon store**, a region per producer. Each piece is an
`OverlayInstance` built from nothing that depends on the camera, and a producer
writes its region only when what it draws from changes. Two of the three
producers are keyed that way: the track rays and the measurements write nothing
on a frame that only moves the camera. The bench figure is the exception, and
this draft is about it.

Amends:
- [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The overlay
  ribbon store": the bench figure's row of the producer table, and § "Depth
  against the scene", which says the figure is rebuilt every frame.

Decided by the maintainer:
- **The bench figure is rebuilt when the camera moves.** Its arrowhead's barbs
  turn to face the eye, so it is rebuilt then. It is a few hundred instances.
  Whether to make the barbs turn in the shader instead is to be judged once the
  rest is built, which it now is.

## The bench figure's barbs

The bench figure's producer rebuilds every frame because its arrowhead's barbs
turn to face the eye. Judge whether the barbs should turn in the shader instead,
so that the figure's key no longer holds the camera's pose and a camera move
writes nothing for it either.

## Steps

1. **The store.** *Built*
   ([gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The
   overlay ribbon store").
2. **Instances independent of the camera.** *Built*
   ([gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) § "The
   overlay ribbon store", [gui/measurement-view.md](../gui/measurement-view.md)
   § "Drawing measurements").
3. **Track rays.** *Built*: the rays are the store's first producer on the
   scene target, anti-aliased strokes held at `BEHIND_OPACITY` where the scene
   hides them, and a ray to a point at infinity ends at a direction drawn to its
   vanishing point ([gui/architecture.md](../gui/architecture.md) § "Track Ray
   Visualization", [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md)
   § "The overlay ribbon store").
4. **The bench figure's barbs:** judge whether they should turn in the shader,
   so the figure's key no longer includes the camera (§ "The bench figure's
   barbs").
