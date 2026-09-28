# Finding the nearby tracks

A person looking at one photograph of a reconstruction can point at a pixel
and ask what the other photographs agree is there, or near there. This
operation answers with **nearby tracks**: 3D points close to the pixel in that
photograph that several photographs see, each one ready to be put on the bench
and worked on. The scene near a pixel can hold surfaces at very different
distances, a tree in front of a building in front of the sky, so the answer is
a set of hypotheses rather than one estimate. The tracks are grouped into
**depth layers**, ranges of distance along the pixel's ray that overlap, and
each layer carries a rank, the evidence behind it, and a confidence that the
pixel is on it. Each track gets a label that names the query, its layer's rank
and its place in the layer. Nothing is written to the reconstruction; a caller
that wants the tracks as points commits them.

The operation combines four pieces, each specified on its own: the
[matching sources](nearby-sources.md) that propose candidates near the pixel,
the [distance range](distance-range.md) of each candidate's sightings and its
class, the [far-field sweep](far-field-sweep.md) of the pixel's own patch, and
the [depth layers](depth-layers.md) that group and rank them. What this spec
adds is the order they run in, when the sources stop, when the far-field
sweep runs, and the tracks and labels the bench takes. It is the anchor finder
of the track-at-pixel harness
([`scripts/track_at_pixel/anchors.py`](../../../scripts/track_at_pixel/anchors.py),
`find_anchors`) moved into core, and it scores the same there. The viewer
calls it from Image Detail's *Find Nearby Tracks* and the wire's
`find_nearby_tracks`, which put every usable track on the bench and commit the
new ones as one version ([the viewer](../../gui/bench.md#find-nearby-tracks)).

## Rust API

In [`bench/nearby/find.rs`](../../../crates/sfmtool-core/src/bench/nearby/find.rs),
re-exported from `sfmtool_core::bench`, and bound as
`sfmtool._sfmtool.bench.find_nearby_tracks`
([`nearby_tracks.rs`](../../../crates/sfmtool-py/src/bench/nearby_tracks.rs)).

```rust
pub fn find_nearby_tracks(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],      // one decoded, posed view per image
    grey: &GreyImages,                 // the blurred grey images, kept across queries
    sources: &NearbyTrackSources<'_>,  // the optional inputs of the sources
    image: u32,
    pixel: [f64; 2],
    options: &NearbyTrackOptions,
    progress: &Progress<'_>,
) -> Result<NearbyTracks, NearbyTracksError>;
// NoSuchImage | PixelOffImage | InputMismatch | RowMismatch
// | NotAMatchingSource | Cancelled

pub struct NearbyTrackSources<'a> {
    pub clusters: Option<&'a MatchesClusters>,      // the clusters source
    pub guided: Option<GuidedSource<'a>>,           // guided matching
    pub sift_index: Option<SiftIndexSource<'a>>,    // the constellation source
}

pub struct NearbyTrackOptions {
    pub sources: Vec<NearbySource>,  // Points, Clusters, Guided, Constellation
    pub stop: StopRule,              // Enough (default) | Never
    pub enough_count: usize,         // 2
    pub enough_px: f64,              // 20
    pub far_field_when: FarFieldWhen, // Needed (default) | Always | Never
    pub points: PointsOptions,
    pub clusters: ClusterTracksOptions,
    pub guided: GuidedOptions,
    pub constellation: ConstellationSeedOptions,
    pub range: RangeOptions,
    pub far_field: FarFieldOptions,
    pub layers: DepthLayerOptions,
    pub tracks: BenchTrackOptions,   // build (true), radius_px (8)
    pub label: Option<String>,       // replaces the group label
}

pub struct NearbyTracks {
    pub group_label: String,          // `<stem>@<x>,<y>`, or the caller's
    pub tracks: Vec<NearbyTrack>,     // every one found, in the order found
    pub layers: Vec<DepthLayer>,      // nearest first; members index `tracks`
    pub report: NearbyTracksReport,
}
impl NearbyTracks {
    pub fn bench_order(&self) -> Vec<usize>;       // the usable ones, in label order
    pub fn first_layer(&self) -> Option<&DepthLayer>; // the rank-1 layer
}

pub struct NearbyTrack {
    pub finding: NearbyFinding,       // Candidate(NearbyCandidate) | FarField(Box<FarFieldReading>)
    pub distance: f64,                // along its pixel's ray; infinite for a bearing
    pub range: [f64; 2],
    pub class: RangeClass,            // bounded, far; usable if either
    pub support: usize,
    pub layer: Option<usize>,         // when usable
    pub order: Option<usize>,         // its place in the layer, 0 first
    pub label: Option<String>,        // when usable
    pub point: Option<u32>,           // the existing point, for Points
    pub track: Option<Result<EditableTrack, String>>, // built, for the rest
}
impl NearbyTrack {
    pub fn source(&self) -> NearbySource;
    pub fn sightings(&self) -> &[(u32, [f64; 2])];
    pub fn query_pixel(&self) -> [f64; 2];
    pub fn distance_px(&self) -> f64;
    pub fn n_views(&self) -> usize;
    pub fn usable(&self) -> bool;
}

pub struct NearbyTracksReport {
    pub sources: Vec<SourceReport>,   // source, found, skipped, seconds, range_seconds
    pub stopped_after: Option<NearbySource>,
    pub far_field: Option<FarFieldRun>, // trigger, report, dropped
    pub layers_seconds: f64,
    pub tracks_seconds: f64,
}

pub fn nearby_group_label(image_stem: &str, pixel: [f64; 2]) -> String;
pub fn nearby_track_label(group: &str, rank: usize, order: usize, point: Option<u32>) -> String;
```

**Why it takes what `build_track_at_pixel` takes.** The two answer questions
about the same pixel from the same inputs, and a caller that holds one query's
inputs holds the other's: the version, the decoded views, the evidence files
built once per capture, the pixel, the options and a progress handle. The grey
images are the one addition, because the far-field sweep and the layers sample
them and the caller keeps them across queries.

**Why every input in `NearbyTrackSources` is optional.** A source whose input
is missing is skipped and named in the report rather than refused: an
`embedded_patches` reconstruction with no `.sift` files still gets its
existing points and the far-field sweep, which need only the reconstruction
and the photographs.

**Why `tracks` holds everything found and not only the usable ones.** The
layers' members, each track's support and the harness's scoring all refer to
every candidate by its place in the order found, and a candidate that is not
usable is still evidence someone looking at the result may want. The usable
ones, the ones the bench takes, are `bench_order()`, in the order of their
labels.

**Why an existing point carries its index and not a track.** A point near the
pixel is already a track with its own frame, bitmap and keypoints; rebuilding
it from its sightings would give a second, worse copy. So a
`NearbySource::Points` track carries `point` and no `track`, and the caller
puts the point's own track on the bench, as *Edit on Bench* does
([`create_track`](editable-track.md)), under the track's label. It is never
committed again.

**Why the labels are built in core.** The Python binding, the viewer and the
wire name the same tracks, and a label built in one place cannot drift between
them.

```rust
use sfmtool_core::bench::{find_nearby_tracks, GreyImages, NearbyTrackOptions, NearbyTrackSources};
use sfmtool_core::progress::Progress;

let grey = GreyImages::new(views.len());
let sources = NearbyTrackSources { clusters: Some(&clusters), ..Default::default() };
let found = find_nearby_tracks(
    &edited, &views, &grey, &sources, image, pixel,
    &NearbyTrackOptions::default(), &Progress::none(),
)?;
for k in found.bench_order() {
    let t = &found.tracks[k];
    match (&t.point, &t.track) {
        (Some(point), _) => { /* put the point's own track on the bench */ }
        (None, Some(Ok(track))) => { /* put `track` on the bench, commit it */ }
        _ => {}
    }
    println!("{}", t.label.as_deref().unwrap());
}
```

## How the pieces combine

**The sources and the stopping rule.** The matching sources run in the order
`options.sources` names them; the default is the reconstruction's own points,
the clusters, guided matching, then the constellation query, strongest first.
Each candidate a source returns gets its distance along its pixel's ray from
the queried camera's centre, its [`distance_range`](distance-range.md) within
`range.tolerance_px`, and its class. After each source, with `StopRule::Enough`,
the query stops when `enough_count` usable candidates lie within `enough_px` of
the pixel: the next source costs more and adds hypotheses the ones found
already settle. `StopRule::Never` runs every source, which is how the harness
measures each one.

**The far-field sweep's trigger.** With `FarFieldWhen::Needed` the sweep runs
after the sources when they leave the pixel's distance open: the usable
candidates fall in no depth layer or in more than one, or none lies within a
pixel of the pixel. The count of layers is the grouping alone, which reads no
photograph. A pixel on the sky or a distant skyline is the case this is for:
the sources find nearer points around it, often one layer of them, and none at
the pixel itself. Each reading keeps the range the sweep gave it, the stretch
between its neighbouring disparities; a reading the sweep's refit moved gets
its sightings' range instead, like a matching source's candidate. `Always`
and `Never` run it or not regardless.

**The layers.** Every track found, the sweep's readings after the sources',
goes to [`depth_layers`](depth-layers.md) with its range and class, and gets
its support back; the usable ones are grouped, read and ranked.

**The tracks for the bench.** With `tracks.build`, every usable track that is
not an existing point is built as a track-stage `EditableTrack`: a cluster
seeded at its sighting in the queried image with a patch of `tracks.radius_px`,
every other sighting added `in` by hand, and the upgrade to the track stage,
which triangulates the sightings and runs the track-stage fit over the frame
it builds. This is the path the far-field sweep's refit and track-at-pixel
build their tracks by (`seed_cluster_with`, `upgrade_sightings`). The track
carries the bench's default thresholds. A track whose build fails carries the
reason and the others are still returned. The builds are independent and run
side by side. Building costs more than finding: on seoul_bull with the default
stopping rule and one thread a query, finding takes about 20 ms a query in the
full pass and 5 in the empty one, and building the tracks another 57 and 75;
with sixteen threads for one query, as the viewer runs it, the building takes
3 to 9 ms. A caller that only reads the layers, like the harness's scoring,
turns it off.

## The labels

A query's tracks share a **group label**, the queried image's stem and the
pixel rounded to whole pixels, `<stem>@<x>,<y>`: the label a cluster seeded at
the pixel gets ([`ClusterSeed::label`](editable-track.md)), so a person who
knows one reads the other. A caller's `options.label` replaces it. Each usable
track's label is the group label, a space, its layer's rank, and a letter for
its place in the layer, `a` first, `z` then `aa`; an existing point adds
` pt <index>`:

    frame_13@412,230 1a
    frame_13@412,230 1b pt 812
    frame_13@412,230 2a

**The rank comes first** because it says which depth the photographs favour,
so the labels sort best-supported first. When the layers are not ranked (the
evidence is off), the layer's place, nearest first, stands in for the rank.

**Within a layer the tracks run by their distance from the pixel, nearest
first**, ties keeping the layer's order. The track nearest the pixel is the
best stand-in for the pixel on that surface and the one a caller walks from,
so `1a` is the track the viewer makes active. The layer's own member order is
by the near end of each range, which says nothing about the pixel.

## Parameters

Defined in `NearbyTrackOptions::default` and `BenchTrackOptions::default` in
[`find.rs`](../../../crates/sfmtool-core/src/bench/nearby/find.rs); the
sections' own defaults are in their specs. The harness names are in
parentheses.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sources` (`sources`) | points, clusters, guided, constellation | The matching sources, in the order they run |
| `stop` (`stop`) | `Enough` | Stop after the first source that leaves enough usable candidates near the pixel, or `Never` |
| `enough_count` (`min_anchors`) | `2` | How many usable candidates are enough |
| `enough_px` (`enough_px`) | `20.0` | How near the pixel, in px, one counts |
| `far_field_when` (`infinity`) | `Needed` | Run the far-field sweep when needed, `Always` or `Never` |
| `tracks.build` | `true` | Build the bench tracks |
| `tracks.radius_px` | `8.0` | Their patch's half-width, in px of the queried image |
| `label` | `None` | The group label, in place of `<stem>@<x>,<y>` |

The reach of "at the pixel" in the far-field trigger is the depth layers'
`AT_PIXEL_PX`, 1 px.

## Python bindings

```python
result = bench.find_nearby_tracks(edited, images, sources, image, pixel,
                                  options={"stop": "never"}, label=None)
edited, result = bench.find_nearby_tracks(..., commit=True)
```

`sources` is the `bench.NearbyTrackSources` the matching sources' bindings
take. `options` overrides the top-level fields by name (`sources` as a list
or `+`-joined, with `tracks` accepted for `points`; `stop`; `enough_count`;
`enough_px`; `far_field_when`) and the sections' fields as
`"<section>.<field>"`, as `build_track_at_pixel` takes them: `points.`,
`clusters.`, `guided.`, `constellation.`, `range.`, `far_field.`, `layers.`
and `tracks.`. An unknown key is a `ValueError`, as is a query that names no
place.

The result is a dict. `tracks` holds the usable tracks in label order, each
with `label`, `source`, `found` (its index into `found`), `layer`, `rank`,
`confidence`, `pixel`, `distance_px`, `range`, `n_views`, `point` and
`track` (an `EditableTrack`, or `None` for an existing point or when building
is off), and `error` when building failed. With `commit=True` every built
track is committed with the bench's commit in label order, its row's `point`
set to the new point, and the call returns `(EditedReconstruction, result)`; a
track the commit refuses gets an `error` and the rest are committed.

For the harness, `found` holds every track as the anchor dicts `find_anchors`
returned (the source's keys, the far-field reading's, `distance`, `range`,
`bounded`, `far`, `support`, and `label`), `layers` the layer dicts
`depth_layers` returns, and `stages` its per-source records (`source` in the
harness's names, `found`, `seconds`, `range_seconds`, `skipped`, and an
`evidence` record for the layers). `report` carries the same in core's names,
with `stopped_after`, `far_field` (`trigger`, `found`, `dropped`, `seconds`)
and the layers' and tracks' seconds. The harness's `find_anchors` calls it by
default (`finder_impl="rust"`) with `tracks.build` off, and returns `found`,
`layers` and `stages` as its `anchors`, `layers` and `stages`;
`finder_impl="python"` runs its own loop over the pieces' bindings, the
reference.

## Testing

[`find_tests.rs`](../../../crates/sfmtool-core/src/bench/nearby/find_tests.rs)
decides the operation on the synthetic capture of a textured plane: a held-out
grid point's pixel gets the eight points around it as existing-point tracks on
one layer, labelled `1a` to `1h` by distance with their points named, the
sources without input skipped and named, and the far-field sweep run because
none sits at the pixel; the stopping rule stops after the points and runs
every source when asked for more than there are; a cluster at the pixel
becomes a fitted track-stage track with the caller's label that commits as a
new point, and is labelled but not built with building off; a plane far past
the cameras' spread, with no sources, gets one far-field reading that keeps
the sweep's range, a far layer and a built track, and nothing when the sweep
is told never to run; the refusals; and the labels' letters and forms.
[`tests/rust_bindings/test_nearby_tracks_rust_bindings.py`](../../../tests/rust_bindings/test_nearby_tracks_rust_bindings.py)
checks on the seoul_bull fixture that every source finds what its own binding
finds, in order, with the range bindings' ranges and classes, the far-field
binding's readings after them, and the depth-layers binding's layers and
support; the labels, their order and the caller's label; that `commit=True`
adds every built track as a point and leaves the given version alone; the
skipped sources; and the options' refusals.

**Parity with the harness.** Run from the harness with `finder_impl=rust` and
`finder_impl=python`, over both ground truths (seoul_bull's 1277 queries and
Kerry Park's 3903), full and empty passes, with every source and with the
default stopping rule, the two return the same anchors with the same sources,
sightings, classes and support, the same layers with the same members, ranks,
evidence, scores, keys and confidences, and identical summaries. The one
difference is in the last bits of a candidate's distance along its pixel's
ray, which the Python loop computes with its own camera and core with its
own, and so of the range ends found from it: at most 3e-14 relative, too small
to move a class, a layer or a reading. The two take the same time within the
run-to-run noise, since the Python loop calls the same core pieces: with the
default stopping rule, about 20 ms a query in seoul_bull's full pass and 80 in
Kerry Park's.

## Non-goals

The harness's `sweep` source (the plane sweep of `candidates/planesweep.py`)
and its older infinity test (`far_test="infinity"`) are not in core; the
far-field sweep replaced the one and the matching sources outrank the other.
The operation does not walk from a nearby track to the pixel, and does not
decide what to do with a low confidence. It commits nothing; the Python
binding's `commit` is the bench's commit applied to each built track, and the
viewer's *Find Nearby Tracks* commits them as one version
([the viewer](../../gui/bench.md#find-nearby-tracks)).
