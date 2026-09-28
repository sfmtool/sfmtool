# Finding the tracks near a pixel: the viewer and the wire

**Status:** Draft. Decided: the operation is built in core and bound to
Python ([core/bench/nearby-tracks.md](../core/bench/nearby-tracks.md)); what
remains is the viewer's entry in Image Detail's context menu after *Edit on
Bench*, the batch commit as one version, and the wire tool, where committing
is an option. The name, the labels, one version per batch and the handling of
existing points are settled as listed under [Decisions](#decisions).

## Purpose

A person looking at one photograph of a reconstruction can point at a pixel and
ask what the other photographs agree is there, or near there. The operation
that answers, `find_nearby_tracks`, returns **nearby tracks**: 3D points close
to the pixel that several photographs see, grouped into ranked **depth
layers**, each with a label and ready for the bench. It is specified in
[core/bench/nearby-tracks.md](../core/bench/nearby-tracks.md), with its pieces
in [far-field-sweep.md](../core/bench/far-field-sweep.md),
[distance-range.md](../core/bench/distance-range.md),
[nearby-sources.md](../core/bench/nearby-sources.md) and
[depth-layers.md](../core/bench/depth-layers.md). This draft proposes how a
person in the viewer, and an agent on the wire, call it.

It is the first step of track-at-pixel as the harness now frames it
([the anchors step](surface-co-solve.md#the-first-step-anchors)): before a
track is built at the pixel, find the geometry around it that the photographs
already agree on.

## The viewer

**Find Nearby Tracks**, in Image Detail's context menu directly after *Edit on
Bench*, runs the operation at the right-clicked pixel as a background
operation, `Find nearby tracks`, with the index files that are current, as
*Create Track Here* does. It is greyed with the same reasons: the node is busy,
the image is not posed. When it lands, every track in the result's
`bench_order()` is put on the bench under its label: an existing point's own
track, as *Edit on Bench* puts it there, and the built track for the rest.
The `1a` track, the nearest the pixel on the rank-1 layer, becomes the active
item, and the tracks that are not existing points are committed as new
points. The status line says how many tracks, in how many layers, and the
first layer's confidence.

## The wire

`find_nearby_tracks` in the bench family: `reconstruction_label`,
`camera_image`, `pixel`, `commit` (default `true`), and an optional `label` for
the group. It runs in the background like `create_track_at_pixel` and answers
with the layers and, per track, its label, point (when committed or existing),
source, layer, rank, confidence, range and pixel in the queried image.

## Decisions

1. **The name.** *Find Nearby Tracks* for the entry, `find_nearby_tracks` in
   core, Python and on the wire, *nearby track* for one result and *depth
   layer* for a group, with a glossary row saying that *anchor* keeps its
   track-at-pixel meaning. Core, Python and the glossary have them.
2. **Labels.** Built in core (`nearby_group_label`, `nearby_track_label`):
   the group label `<stem>@<x>,<y>` and per track its layer's rank and a letter
   for its place in the layer by distance from the pixel, `frame_13@412,230
   1a`, with ` pt <index>` for an existing point. The wire's `label` replaces
   the group label, as the core option does.
3. **One version for the batch.** A menu click that commits eight points is
   one undo: the viewer puts the tracks on the bench and commits them in one
   version, *Found 8 nearby tracks at frame_13@412,230*, through a batch path in
   `AppState` beside `commit_bench_track`, and a `Finished::NearbyTracks`
   beside `Finished::TrackAtPixel`.
4. **Existing points.** The reconstruction's own points near the pixel are the
   strongest source. Each comes back as a nearby track carrying its point and
   no built track: its own track is put on the bench, as *Edit on Bench* would,
   under its label (`frame_13@412,230 1a pt 812`), and it is never committed
   again.

## Stages

The operation was built in stages, each a public function measured against
the harness before the next built on it; all five are built and their standing
specs are linked above. The harness's `find_anchors` calls the combined
operation by default (`finder_impl="rust"`) and keeps its Python loop as the
reference until that is retired.

6. **The viewer and the wire.** *Find Nearby Tracks* in Image Detail's
   context menu, the batch commit as one version, `Finished::NearbyTracks`,
   and the `find_nearby_tracks` wire tool with `commit`.
