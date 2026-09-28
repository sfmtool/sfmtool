# The matching sources near a pixel

A person looking at one photograph of a reconstruction can point at a pixel
and ask what the other photographs agree is there, or near there. Before that
can be answered, something has to propose candidates: points close to the
pixel in that photograph that several photographs see, each with where every
photograph sees it. The **matching sources** do this, each from a different
kind of evidence: the reconstruction's own points, and later the clusters of
matched keypoints, keypoints matched along their rays, and the SIFT index's
constellation query. Every source returns the same shape, a **nearby
candidate**: a 3D point, its sightings (the queried photograph's first), how
far it sits from the pixel, and how well its sightings meet. Nothing is
written to the reconstruction.

The sources are the first step of the anchor finder in the track-at-pixel
harness
([`scripts/track_at_pixel/anchors.py`](../../../scripts/track_at_pixel/anchors.py),
`from_tracks`), where they were designed and measured, and the third part of
that finder moved into core ([the plan](../../drafts/nearby-tracks.md)). A
candidate is a hypothesis, not a decision: near a pixel the scene can hold
surfaces at very different depths, and grouping the candidates by the
[distance ranges](distance-range.md) their sightings allow is a later step.

## Rust API

The sources live in
[`bench/nearby/`](../../../crates/sfmtool-core/src/bench/nearby.rs), one file
each, with the shared shape in
[`candidate.rs`](../../../crates/sfmtool-core/src/bench/nearby/candidate.rs),
all re-exported from `sfmtool_core::bench`. The points source is
[`points.rs`](../../../crates/sfmtool-core/src/bench/nearby/points.rs), bound as
`sfmtool._sfmtool.bench.nearby_points`.

```rust
pub fn nearby_points(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    image: u32,
    pixel: [f64; 2],
    options: &PointsOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError>;

pub struct NearbyCandidate {
    pub source: NearbySource,            // Points
    pub id: Option<u32>,                 // the point
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

pub enum NearbySourceError { NoSuchImage { .. }, PixelOffImage { .. }, InputMismatch { .. } }
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
cameras.

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
different order.

## Parameters

Each source's options are a struct whose defaults are the harness's
`DEFAULTS`, named without the flat dictionary's prefix.

| Parameter | Default | Harness | Meaning |
|-----------|---------|---------|---------|
| `PointsOptions::radius_px` | `40.0` | `track_radius_px` | how far from the pixel a point's observation may be |
| `PointsOptions::max_points` | `8` | `track_max` | the most points kept, the nearest |
| `PointsOptions::min_views` | `2` | `track_min_views` | the fewest observations a point needs |
| `PointsOptions::max_reproj_px` | `2.0` | `max_reproj_px` | the largest error any observation may have |

## Python bindings

```python
from sfmtool._sfmtool import bench

anchors = bench.nearby_points(edited, images, image, (x, y),
                              options={"radius_px": 40.0, "max_points": 8})
```

`images` is a list of decoded images or an `ImagePyramidSet`, as every bench
step takes them. `options` overrides fields by their Rust names; an unknown key
is a `ValueError`, as is an image or pixel that names no place. Each candidate
comes back as the harness's anchor dict: `source` (`"tracks"`, the harness's
name for the points), `id`, `position`, `views` (`[image, x, y]` rows, the
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
that names no place is refused.
[`tests/rust_bindings/test_nearby_sources_rust_bindings.py`](../../../tests/rust_bindings/test_nearby_sources_rust_bindings.py)
checks the bindings on the seoul_bull fixture with its longest track held out.
Parity with the harness's Python is measured by running the harness with
`sources_impl=rust` and `sources_impl=python`.
