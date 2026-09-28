# Finding the tracks near a pixel

**Status:** Draft. Decided: the operation is a port of the anchor finder in
[`scripts/track_at_pixel/anchors.py`](../../scripts/track_at_pixel/anchors.py)
into `sfmtool-core`, with the same sources, far-field sweep, depth layers,
ranking and confidence, exposed to Python, to the viewer's Image Detail context
menu after *Edit on Bench*, and to the wire, where committing is an option. It
is built in [stages](#stages), each a public function measured against the
harness before the next builds on it, starting with the far-field sweep.
The name, the labels, one version per batch and the handling of existing
points are settled as listed under [Decisions](#decisions).

## Purpose

A person looking at one photograph of a reconstruction can point at a pixel and
ask what the other photographs agree is there, or near there. This operation
answers with **nearby tracks**: 3D points close to the pixel in that photograph
that several photographs see, each a track ready for the bench. The scene near
a pixel can hold surfaces at very different depths, a tree in front of a
building in front of the sky, so the answer is a set of hypotheses rather than
one estimate. The tracks are grouped into **depth layers**, ranges of distance
along the pixel's ray that overlap, and each layer carries a rank, the evidence
behind it, and a confidence that the pixel is on it. Walking from a nearby
track to the pixel itself, and deciding what to do with a low confidence, are
later steps and not part of this operation.

It is the first step of track-at-pixel as the harness now frames it
([the anchors step](surface-co-solve.md#the-first-step-anchors)): before a
track is built at the pixel, find the geometry around it that the photographs
already agree on. Every threshold and constant comes from that harness, where
the Python finder was measured against two ground truths; the port is done when
the harness, calling the Rust operation, scores the same.

## Rust interface

```rust
pub fn find_nearby_tracks(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    sources: &NearbyTrackSources<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &NearbyTrackOptions,
    progress: &Progress<'_>,
) -> Result<NearbyTracks, NearbyTracksError>;
```

It sits in `sfmtool_core::bench` beside `build_track_at_pixel`, takes the same
inputs in the same shape, and writes nothing: the tracks it returns are bench
items like any other. `NearbyTrackSources` carries the optional inputs, each of
which enables a source when present: the SIFT index (`forest`, `keypoints`) for
the constellation query and guided matching, the `.sift` descriptors for guided
matching, and the cluster-patches `MatchesClusters` for the clusters. A source
whose input is missing is skipped and named in the report, not an error, so
an embedded-patches reconstruction with no `.sift` files still gets its
existing points, and the far-field sweep, which needs only the photographs.

```rust
pub struct NearbyTracks {
    pub tracks: Vec<NearbyTrack>,  // usable ones only, in layer rank order
    pub layers: Vec<DepthLayer>,   // nearest first; `rank` orders them
    pub report: NearbyTracksReport, // per source: found, skipped and why, seconds
}

pub struct NearbyTrack {
    pub track: EditableTrack,      // track stage, fitted
    pub source: NearbySource,      // Points | Clusters | Guided | Constellation | FarField
    pub pixel: [f64; 2],           // where it sits in the queried image
    pub distance_px: f64,          // from the queried pixel
    pub range: [f64; 2],           // distances along its pixel's ray its views allow
    pub layer: usize,              // index into `layers`
    pub point: Option<u32>,        // the existing point it is, for `Points`
    pub far_field: Option<FarFieldReading>, // the sweep's metrics
}

pub struct DepthLayer {
    pub range: [f64; 2],
    pub tracks: Vec<usize>,        // indexes into `tracks`
    pub evidence: LayerEvidence,   // votes, support, patch reads, nearest_px, ...
    pub score: f64,
    pub key: f64,
    pub rank: usize,               // 1 is the best-supported
    pub confidence: f64,           // that the pixel is on this layer
}
```

`NearbyTrackOptions` holds the harness's `DEFAULTS`, one field each, with the
same names without the prefixes the flat Python dictionary needed (`ff_wide`
becomes `far_field.wide`), and the Python binding takes the same
`"section.field"` overrides `build_track_at_pixel`'s does.

**Why the result keeps the layers beside the tracks.** A caller that walks to
the pixel wants the ranked layers; a caller that only wants points, the
viewer's menu entry, wants the tracks. Both are the same computation.

**Why tracks and not anchors.** The finder's result was called anchors in the
harness, but *anchor* already means the anchored fit in
[track-at-pixel](../core/bench/track-at-pixel.md), the slide of a patch onto the
pixel. What the operation returns are tracks, near the pixel, and the bench
already knows what a track is.

## Python

```python
result = bench.find_nearby_tracks(edited, images, sources, image, pixel,
                                  options={...}, commit=False)
# result.tracks, result.layers, result.report
edited, result = bench.find_nearby_tracks(..., commit=True)  # also returns the new points
```

`sources` is `NearbyTrackSources(edited, forest, keypoints, matches, sift)`,
built like `TrackAtPixelSources`. With `commit=True` every returned track that
is not an existing point is committed with `bench.commit`, and each
`NearbyTrack` carries the point it became. The harness's `--mode anchors` then
calls this in place of `anchors.find_anchors`, and `score_anchors` scores its
result.

## The viewer

**Find Nearby Tracks**, in Image Detail's context menu directly after *Edit on
Bench*, runs the operation at the right-clicked pixel as a background
operation, `Find nearby tracks`, with the index files that are current, as
*Create Track Here* does. It is greyed with the same reasons: the node is busy,
the image is not posed. When it lands, every returned track is put on the bench
under a label that names the group (below), the rank-1 layer's nearest track
becomes the active item, and the tracks that are not existing points are
committed as new points. The status line says how many tracks, in how many
layers, and the first layer's confidence.

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
   track-at-pixel meaning.
2. **Labels.** One group label from the query, the `ClusterSeed` form
   `<stem>@<x>,<y>`, and one short suffix per track naming its layer rank and
   its order in the layer: `frame_13@412,230 1a`, `frame_13@412,230 1b`,
   `frame_13@412,230 2a`. The rank says which is the best-supported depth, and
   the shared prefix keeps a group together in the Scene tree. The wire's
   `label` replaces the prefix.
3. **One version for the batch.** A menu click that commits eight points is
   one undo: the viewer puts the tracks on the bench and commits them in one
   version, *Found 8 nearby tracks at frame_13@412,230*, through a batch path in
   `AppState` beside `commit_bench_track`, and a `Finished::NearbyTracks`
   beside `Finished::TrackAtPixel`.
4. **Existing points.** The reconstruction's own points near the pixel are the
   strongest source. Each is returned as a nearby track seated on its point:
   its own track is put on the bench, as *Edit on Bench* would, under the
   group's label with the point named in the suffix
   (`frame_13@412,230 1a pt 812`), and it is never committed again.

## Implementation notes

**What is ported and what exists.** The track side exists in core: the
neighbourhood queries (`ObservationIndex`, `MatchesClusters::near`), the
constellation seed, building a track from sightings (`track_from_sightings` in
`track_at_pixel/finish.rs`, to be shared), the fit and the commit. The rest is
Python only and is ported: triangulation with per-view errors and the
cluster vetting that drops the worst member; guided matching (keypoint rays
from each image, closest approach to the query keypoint's ray, the ratio test
and the two passes); `distance_range`; the patch read on a plane facing the
queried camera, whole and middle ZNCC from the same samples, on the blurred
grey images the harness samples; the far-field sweep with its peaks,
prominence and metrics; the pairwise grouping by average linkage and the
relocation fit; the layers, their evidence reads, the key and the confidence.
The camera needs projecting a direction (`w = 0`), which `ViewCamera` lacks.

**Parity.** The harness keeps its Python finder as the reference until the Rust
one matches it on both ground truths (layer, rank-1 right, confidence AUC, and
per source the counts in the summary), then retires it.

**The pairwise table is O(N²)** in the images of a far-field reading, capped at
16; how it scales past that is open.

## Stages

The operation is built from public functions, each useful and measurable on
its own, then combined. Every stage is one pull request that adds a core
function, its Python binding, and a change to the harness that calls the Rust
function in place of the Python one and shows the scores unchanged on both
ground truths. `anchors.py` shrinks stage by stage until only the harness's
scoring is left.

1. **The far-field sweep.** `far_field_sweep(edited, views, image, pixel,
   options) -> Vec<FarFieldReading>`: the pixel's patch read from infinity in
   to 16 px of disparity in every image it lands in, one reading per peak, each
   with its range and metrics, its images grouped by pairwise middle ZNCC, and
   the relocation fit when the query stands apart. It brings with it the two
   pieces it is made of, both public because the later stages use them:
   - **Reading a patch along a ray**, `read_patch_along_ray`: the pixel's
     patch sampled on a plane facing the queried camera at each of a list of
     distances, in each of a list of images, giving the whole and the middle
     ZNCC from the same samples, and the samples when asked. The grey images
     are the harness's, blurred once per image and cached with the views.
   - **Projecting a direction**, a `w = 0` projection on the camera the
     finder uses, which the far field and the ranges need.
2. **Ranges.** `distance_range(views, image, pixel, sightings, distance,
   tolerance_px) -> [f64; 2]`: the distances along the pixel's ray at which
   every sighting stays within its tolerance, which turns any set of sightings
   into a range that can be compared with another.
3. **The matching sources.** Each returns candidate tracks with their
   sightings, ranges and triangulation errors:
   - `nearby_points`: the reconstruction's own points observed near the
     pixel, with the reprojection check.
   - `nearby_cluster_tracks`: the cluster-patches clusters near the pixel,
     vetted by triangulating their members and dropping the worst.
   - `guided_matches`: the keypoints near the pixel matched along the rays of
     every other image's keypoints, with the ratio test and the two passes.
   - The constellation seed, from the existing `seed_cluster`.
4. **Depth layers.** `depth_layers(views, image, pixel, candidates, options)
   -> Vec<DepthLayer>`: the usable candidates grouped by overlapping ranges,
   each layer's evidence read with `read_patch_along_ray`, its key, rank and
   confidence.
5. **Finding the nearby tracks.** `find_nearby_tracks`, the interface above:
   the sources in order with the stopping rule, the far-field sweep when they
   leave the depth open, the layers, and the tracks built and fitted for the
   bench. Its Python binding replaces `anchors.find_anchors` in the harness.
6. **The viewer and the wire.** *Find Nearby Tracks* in Image Detail's
   context menu, the batch commit as one version, `Finished::NearbyTracks`,
   and the `find_nearby_tracks` wire tool with `commit`.

Stages 2 and 3 do not depend on stage 1 and could go in either order; stage 4
needs the patch read from stage 1 and the ranges from stage 2.
