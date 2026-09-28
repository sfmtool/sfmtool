# The matching sources near a pixel

A person looking at one photograph of a reconstruction can point at a pixel
and ask what the other photographs agree is there, or near there. Before that
can be answered, something has to propose candidates: points close to the
pixel in that photograph that several photographs see, each with where every
photograph sees it. The **matching sources** do this, each from a different
kind of evidence: the reconstruction's own points, the clusters of matched
keypoints, keypoints matched along their rays, and later the SIFT index's
constellation query. Every source returns the same shape, a **nearby
candidate**: a 3D point, its sightings (the queried photograph's first), how
far it sits from the pixel, and how well its sightings meet. Nothing is
written to the reconstruction.

The sources are the first step of the anchor finder in the track-at-pixel
harness
([`scripts/track_at_pixel/anchors.py`](../../../scripts/track_at_pixel/anchors.py),
`from_tracks`, `from_clusters`, `from_guided`), where they were designed and measured, and the third part of
that finder moved into core ([the plan](../../drafts/nearby-tracks.md)). A
candidate is a hypothesis, not a decision: near a pixel the scene can hold
surfaces at very different depths, and grouping the candidates by the
[distance ranges](distance-range.md) their sightings allow is a later step.

## Rust API

The sources live in
[`bench/nearby/`](../../../crates/sfmtool-core/src/bench/nearby.rs), one file
each, with the shared shape in
[`candidate.rs`](../../../crates/sfmtool-core/src/bench/nearby/candidate.rs),
all re-exported from `sfmtool_core::bench`: the points in
[`points.rs`](../../../crates/sfmtool-core/src/bench/nearby/points.rs), the
clusters in
[`clusters.rs`](../../../crates/sfmtool-core/src/bench/nearby/clusters.rs) and
guided matching in
[`guided.rs`](../../../crates/sfmtool-core/src/bench/nearby/guided.rs), bound as
`sfmtool._sfmtool.bench.nearby_points`, `nearby_cluster_tracks` and
`guided_matches`.
The triangulation the sources share is
[`triangulate.rs`](../../../crates/sfmtool-core/src/bench/nearby/triangulate.rs),
public as `triangulate_sightings`.

```rust
pub fn nearby_points(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    image: u32,
    pixel: [f64; 2],
    options: &PointsOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError>;

pub fn nearby_cluster_tracks(
    views: &[ProjectedImage<'_>],
    clusters: &MatchesClusters,          // the cluster-patches .matches, indexed onto the images
    image: u32,
    pixel: [f64; 2],
    options: &ClusterTracksOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError>;

pub fn guided_matches(
    views: &[ProjectedImage<'_>],
    source: &GuidedSource<'_>,           // keypoints, descriptors, rays
    image: u32,
    pixel: [f64; 2],
    options: &GuidedOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError>;

pub struct GuidedSource<'a> {
    pub keypoints: &'a [ImageKeypoints],     // every image's .sift keypoints
    pub descriptors: &'a [ImageDescriptors], // the same images' .sift descriptors
    pub rays: &'a KeypointRays,              // camera-frame rays, built on first read
}

pub fn triangulate_sightings(views: &[ProjectedImage<'_>], sightings: &[(u32, [f64; 2])])
    -> Option<RayMeeting>;               // { position, errors_px }

pub struct NearbyCandidate {
    pub source: NearbySource,            // Points | Clusters | Guided
    pub id: Option<u32>,                 // the point, the cluster, the keypoint row
    pub position: Vector3<f64>,          // world coordinates
    pub sightings: Vec<(u32, [f64; 2])>, // (image, pixel), the queried image's first
    pub errors_px: Vec<f64>,             // each sighting's reprojection error
    pub query_pixel: [f64; 2],           // its sighting in the queried image
    pub distance_px: f64,                // from the pixel asked about
    pub max_reproj_px: f64,
    pub max_ray_angle_deg: f64,          // widest angle between two sightings' rays
    pub depth: f64,                      // along the queried camera's axis
    pub point: Option<u32>,              // the reconstruction's point, for Points
}

impl NearbyCandidate {
    pub fn n_views(&self) -> usize;
    pub fn image(&self) -> u32;                                  // the queried image
    pub fn ray_distance(&self, views: &[ProjectedImage<'_>]) -> f64;
    pub fn range(&self, views: &[ProjectedImage<'_>], tolerance_px: f64)
        -> Result<[f64; 2], DistanceRangeError>;
}

pub enum NearbySourceError {
    NoSuchImage { .. }, PixelOffImage { .. }, InputMismatch { .. }, RowMismatch { .. },
}
```

**Why one shape for every source.** The step after the sources compares
candidates from all of them: it asks which distances along the pixel's ray
each allows and which agree. It needs the same few things from each, and
nothing a source knows beyond them, so the sources meet in one struct rather
than one type each. `id` is what the source names a candidate by, and `point`
is kept apart from it because only an existing point is one the bench must not
commit again.

**Why the queried image's sighting first.** A candidate becomes a bench track
whose observation 0 is the queried pixel's, and the range is measured along
that sighting's ray, so the first sighting is where both look. A source that
finds sightings in another order moves the queried one to the front, with its
error.

**Why the range is a method and not a field.** A range is the
[distance range](distance-range.md) of the sightings along the query pixel's
ray, at the distance the candidate's point is along it (`ray_distance`), and
its classification depends on the reconstruction's camera spread and the
finder's thresholds. `range` is the one call that turns a candidate into
`distance_range`'s arguments, so the caller that compares candidates computes
the ranges when it needs them.

**Why each source takes exactly its inputs.** A source whose input is missing
(no `.matches` clusters, no `.sift` files) is skipped, not refused, by whoever
combines the sources. Taking each input as a plain reference makes a missing
one impossible to pass, and leaves the skipping, and saying so, to that
caller. The points source reads only the reconstruction and the views'
cameras, the clusters source only the clusters and the cameras.

**Why guided matching reads the `.sift` descriptors.** Guided matching compares
the descriptor of each keypoint near the pixel with those of the keypoints in
every other image whose rays pass close to its own, which are spread over the
whole of each image and change with every query. The SIFT index holds the same
descriptors, but it stores them by corpus feature, in its own order, and
reaching one image's rows through it means a pass over the whole origin table
per image (`FeatureSources::image_feature_ids`), for every image, on every
query; an index may also hold only part of an image's features, which would
make the matches depend on how the index was built rather than on the
photographs. The `.sift` files hold each image's descriptors row for row with
the keypoints every source already reads, so `ImageDescriptors` is read from
them once per capture and kept, 128 bytes a keypoint, as `ImageKeypoints` is.
The rays through the keypoints are kept too: in the camera's frame they depend
on the lens and not the pose, so `KeypointRays` builds each image's the first
time it is read and a query turns them into the world frame with the pose it
holds, rather than undistorting every keypoint of every image per query.

**Why the triangulation is public.** Every source but the points builds its
candidate by meeting rays and reading how far each sighting is from the point,
and the step that joins the sources' sightings into a track does the same.
`triangulate_sightings` is that one computation, returning the errors with the
point because every caller judges the point by them.

```rust
use sfmtool_core::bench::{nearby_points, PointsOptions};

let found = nearby_points(&edited, &views, image, pixel, &PointsOptions::default())?;
for c in &found {
    let range = c.range(&views, 1.0)?;
    println!("{} {:?}: {} views, {:.1} px away, {:?}",
             c.source, c.id, c.n_views(), c.distance_px, range);
}
```

## The sources

### The reconstruction's points

The points observed within `radius_px` (40) of the pixel in the queried
image, nearest first, are the strongest candidates: a solver already agreed on
them. A point is kept when it is finite, has `min_views` (2) or more
observations, and every observation lies within `max_reproj_px` (2) of where
the point projects in its image; at most `max_points` (8) are kept, the
nearest. The candidate's sightings are the point's observations, its first in
the queried image leading, and it sits at that observation's keypoint. The
points are read through the `EditedReconstruction` the caller holds, so a
point the version has deleted, the harness's held-out one, is never found.

### The clusters

A cluster-patches `.matches` file groups every image's keypoints that the SIFT
index matched into clusters, and refines each cluster's members against a
reference, marking each kept or rejected. The clusters with a member within
`radius_px` (48) of the pixel in the queried image, nearest first, up to
`max_clusters` (16), are vetted as candidates. The member in the queried image
is the cluster's nearest there, and must be one the `members` policy admits:
`any` (the default) admits the reference, the kept, the rejected for a low
ZNCC or a large shift, and the unevaluated, letting the triangulation drop the
bad ones; `kept` admits only the reference and the kept. In every other image
the cluster contributes one admitted member, the reference or a kept one
first, then the one whose ZNCC against the reference is highest, in the order
the images are first met in the cluster. The members are then triangulated,
dropping the worst (below). The candidate names its cluster in `id` and sits at
its queried member.

### Guided matching

The keypoints of the queried image within `radius_px` (24) of the pixel and
further than `skip_px` (none by default), up to `max_keypoints` (8), nearest
first, are each matched in every other image. With the cameras posed, a
keypoint's match lies on a keypoint whose ray passes close to its own: for
each keypoint of the other image, the two rays' closest approach is found, and
the keypoint is a candidate when that point is in front of both cameras and
the gap between the rays there, in px of either image (the gap times the focal
length over the distance along the ray), is at most `epipolar_px` (2). Among
the candidates the nearest descriptor is the match when its distance is at
most `ratio` (0.8) of the second nearest's (a lone candidate passes) and at most
`max_distance` (250). Each keypoint's matches are triangulated with it,
dropping the worst (below). A match that passed the ratio test and is further
than `max_distance` but within `loose_distance` (400) is then added, the lowest
ratio first, when the triangulation with it still meets every sighting within
`max_reproj_px`: a match the descriptor alone would not accept is accepted
when the geometry agrees with it. A keypoint with `min_views` (2) or more
sightings is a candidate, named by its row and sitting at the keypoint.

The descriptor distance is Euclidean, computed as the harness computes it in
32-bit floats: the sum of squared differences of two 128-byte descriptors is an
integer below `2^24`, so it is exact, and only the square root rounds. The
ratio and the two distance bars are compared in 32 bits as well, as NumPy
compares a 32-bit value with a Python float.

### Triangulating and dropping the worst

The sightings' rays are met in the least-squares sense: the point minimising
the sum of squared distances to the rays, the solution of
`Σ (I − d dᵀ) x = Σ (I − d dᵀ) c` over each sighting's camera centre `c` and
ray `d`. There is no point when the system is singular (all rays parallel), or
when it lies behind a sighting's camera or outside its lens. Each sighting's
error is the pixel distance from where the point projects in its image. A set
of sightings is accepted when every error is within `max_reproj_px`; otherwise
the worst sighting, the first of the largest, is dropped and the rest are
met again, while three or more remain. The queried sighting is never dropped:
when it is the worst, the set gives no candidate, since a point the queried
image disagrees with is not near the pixel.

## Implementation notes

**The projection is the harness camera's.** Errors are measured with
`ViewCamera::project_homogeneous`, under which a wide-angle lens sees past 90
degrees off its axis, as [distance ranges](distance-range.md) are; a point a
camera cannot see has an infinite error there.

**Parity with the harness.** The harness selects the sources' implementation
with `sources_impl` (`"rust"` by default, `"python"` for the reference). On
both ground truths, full and empty passes, every source run, the harness's
scores are identical. The points source returns the same points with the same
sightings for all but 6 of 27 906 candidates: in Kerry Park two points share a
keypoint, so they are the same distance from the pixel, and the nearest-first
order of a tie follows the spatial index, which the harness builds over every
point and core over the version's live ones; where the cap falls between the
two, each keeps a different one. The depth, the errors and the ray angle
differ from the Python's in the last bits, from the rotation matrix being
built from the quaternion by a different formula and the products summed in a
different order. The clusters source returns the same clusters with the same members
for every query. Their points differ from the Python's in the eighth
significant digit or later, and in the sixth for the worst, two rays 0.003
degrees apart meeting 9 km out: the least-squares system of nearly parallel
rays is ill-conditioned, and the harness solves it with LAPACK and core with
nalgebra's LU. The layers' keys and confidences differ by as little in turn.
Guided matching returns the same keypoints with the same sightings. Two
keypoints of the queried image can share a position, as SIFT's do when it
finds two orientations at one place; they are equally near the pixel, and
both implementations keep them in row order. The harness's reference sorted
them with NumPy's default `argsort`, whose order of ties is not stable and
depends on the processor's vector instructions, so where the cap of eight fell
between two such keypoints it could keep either. It now sorts stably, which
left seoul_bull's scores as they were and changed Kerry Park's by one query:
the full pass's supported-layer shares of clusters and guided by 0.001, and
the rank-1 line's queries with several layers from 2509 to 2508 (1908 to 1907
in the empty pass).

## Parameters

Each source's options are a struct whose defaults are the harness's
`DEFAULTS`, named without the flat dictionary's prefix.

| Parameter | Default | Harness | Meaning |
|-----------|---------|---------|---------|
| `PointsOptions::radius_px` | `40.0` | `track_radius_px` | how far from the pixel a point's observation may be |
| `PointsOptions::max_points` | `8` | `track_max` | the most points kept, the nearest |
| `PointsOptions::min_views` | `2` | `track_min_views` | the fewest observations a point needs |
| `PointsOptions::max_reproj_px` | `2.0` | `max_reproj_px` | the largest error any observation may have |
| `ClusterTracksOptions::radius_px` | `48.0` | `cluster_radius_px` | how far from the pixel a cluster's member may be |
| `ClusterTracksOptions::max_clusters` | `16` | `cluster_max` | the most clusters tried, the nearest |
| `ClusterTracksOptions::max_reproj_px` | `2.0` | `max_reproj_px` | the largest error any member may have |
| `ClusterTracksOptions::members` | `Any` | `cluster_members` | which members may be used, `Any` or `Kept` |
| `GuidedOptions::radius_px` | `24.0` | `guided_radius_px` | how far from the pixel a queried keypoint may be |
| `GuidedOptions::max_keypoints` | `8` | `guided_max` | the most keypoints matched, the nearest |
| `GuidedOptions::skip_px` | `-1.0` | `guided_skip_px` | keypoints this near the pixel or nearer are skipped |
| `GuidedOptions::epipolar_px` | `2.0` | `guided_epipolar_px` | how close a candidate's ray must pass, in px |
| `GuidedOptions::ratio` | `0.8` | `guided_ratio` | the nearest descriptor over the second nearest, at most |
| `GuidedOptions::max_distance` | `250.0` | `guided_max_dist` | the largest descriptor distance of a match |
| `GuidedOptions::loose_distance` | `400.0` | `guided_loose_dist` | the largest of a match added after the triangulation |
| `GuidedOptions::min_views` | `2` | `guided_min_views` | the fewest sightings a candidate needs |
| `GuidedOptions::max_reproj_px` | `2.0` | `max_reproj_px` | the largest error any sighting may have |

## Python bindings

```python
from sfmtool._sfmtool import bench

anchors = bench.nearby_points(edited, images, image, (x, y),
                              options={"radius_px": 40.0, "max_points": 8})

sources = bench.NearbyTrackSources(edited, matches=MatchesFile(path),
                                   sift=[...])  # one .sift path per image
anchors += bench.nearby_cluster_tracks(edited, images, sources, image, (x, y),
                                       options={"members": "any"})
anchors += bench.guided_matches(edited, images, sources, image, (x, y))
```

`NearbyTrackSources` holds the inputs the sources read beside the
reconstruction, each optional and built once per capture: `matches`, a
cluster-patches `MatchesFile`, indexed onto the reconstruction's images by name;
`keypoints`, one `(positions, affine_shapes)` pair of float32 arrays per image
as `TrackAtPixelSources` takes them; and `sift`, one `.sift` path per image,
whose descriptors are read in parallel, and whose keypoints are read too when
`keypoints` is left out. A source whose input it does not hold returns an empty
list. `images` is a list of decoded images or an `ImagePyramidSet`, as every bench
step takes them. `options` overrides fields by their Rust names; an unknown key
is a `ValueError`, as is an image or pixel that names no place. Each candidate
comes back as the harness's anchor dict: `source` (the harness's names,
`"tracks"` for the points, `"clusters"` and `"guided"`), `id`, `position`, `views` (`[image, x, y]` rows, the
queried image first), `query_pixel`, `distance_px`, `n_views`,
`max_reproj_px`, `max_ray_angle_deg` and `depth`.

## Testing

[`bench/nearby/source_tests.rs`](../../../crates/sfmtool-core/src/bench/nearby/source_tests.rs)
decides the sources on the bench's synthetic capture, a grid of points on a
textured plane seen by three cameras, with the middle point deleted and the
query made at its pixel: the eight grid neighbours are found nearest first,
without the deleted point, with the queried sighting first and a range that
holds their distance; a point one photograph puts 3 px away is left out until
the bar allows it; a point with too few observations is left out; and a query
that names no place is refused. On the same plane seen by four cameras, a
cluster with one member 8 px off keeps the other three, and all four once the
bar allows the error; a cluster whose worst member is the queried one gives
nothing; the member policy decides between a kept and a rejected member in
one image; and sightings meet where their rays do, the off one carrying the
largest error. Guided matching, on keypoints placed at the projections of
plane points with their own descriptors, finds the match along the ray in two
images and the loose one in a third after them; a keypoint on the queried ray
further out whose descriptor is nearly as close makes that image's match
ambiguous, and it is refused until the ratio allows it; and descriptors that
are not row for row with the keypoints are refused.
[`tests/rust_bindings/test_nearby_sources_rust_bindings.py`](../../../tests/rust_bindings/test_nearby_sources_rust_bindings.py)
checks the bindings on the seoul_bull fixture with its longest track held out.
Parity with the harness's Python is measured by running the harness with
`sources_impl=rust` and `sources_impl=python`.
