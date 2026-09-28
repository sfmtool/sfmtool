# Distance ranges

A point seen in several photographs is placed where their rays meet, but the
rays never meet exactly, and how far along one photograph's ray the point could
be depends on how far apart the photographs were taken. Two photographs a metre
apart pin down a point four metres away to a few centimetres; two taken a step
apart cannot tell a point a hundred metres away from one at infinity. A
**distance range** says this for one pixel: it is the stretch of the pixel's ray
along which the point can move while every other photograph still sees it
within a pixel of where it was found. A range lets two readings of the same
pixel, from different sources and different photographs, be compared: they
agree when their ranges overlap. What the range says about the point is
summarised in two words: **bounded**, when the photographs pin the distance
down, and **far**, when they put it too far out to tell from infinity.

The ranges are part of the anchor finder in the track-at-pixel harness
([`scripts/track_at_pixel/anchors.py`](../../../scripts/track_at_pixel/anchors.py),
`distance_range`), where every anchor carries one, and the second part of that
finder moved into core ([the plan](../../drafts/nearby-tracks.md)).

## Rust API

In [`bench/nearby/range.rs`](../../../crates/sfmtool-core/src/bench/nearby/range.rs),
re-exported from `sfmtool_core::bench`, and bound as
`sfmtool._sfmtool.bench.distance_range`, `camera_spread` and `classify_range`.

```rust
pub fn distance_range(
    views: &[ProjectedImage<'_>],
    image: u32,
    pixel: [f64; 2],
    sightings: &[(u32, [f64; 2])], // (image, pixel) pairs of the point
    distance: f64,                 // along the unit ray; f64::INFINITY at infinity
    tolerance_px: f64,
) -> Result<[f64; 2], DistanceRangeError>; // [near, far], far may be infinite

pub fn camera_spread(views: &[ProjectedImage<'_>]) -> f64;

pub fn classify_range(range: [f64; 2], camera_spread: f64, options: &RangeOptions)
    -> RangeClass;

pub struct RangeOptions { pub tolerance_px: f64, pub max_span: f64, pub far_spread: f64 }
pub struct RangeClass { pub bounded: bool, pub far: bool } // .usable(): either
```

**Why the views and not a reconstruction.** A range reads only cameras and
poses, but every caller in the bench already holds the views, one per image,
as `far_field_sweep` and `build_track_at_pixel` take them; taking the same
slice keeps one shape of input across the finder's pieces and lets a caller
build the cameras once. The pixels of the views are not read.

**Why sightings as `(image, pixel)` pairs.** That is what every source of the
finder produces, a track's observations as much as a far-field reading's
views, and nothing else about a sighting enters the range. A sighting in the
queried image itself is skipped rather than refused, because the sources list
the query among their sightings.

**Why the distance is an argument.** The range is found by searching outward
from where the point was triangulated, and the error there sets the tolerance.
It is the distance along the pixel's unit ray from the queried camera's centre,
not the depth along the camera's axis; a caller with a point `x` computes it as
`(x - centre) · ray`.

**Why the classification is separate.** The range is a property of the
sightings; bounded and far also depend on the reconstruction's scale, through
the camera spread, and on thresholds the finder tunes, and a caller that
adjusts a range (the harness lowers a far-field reading's near end to where
its patch reading starts to fall) classifies the adjusted one. The camera
spread is computed once per reconstruction and passed in.

```rust
use sfmtool_core::bench::{camera_spread, classify_range, distance_range, RangeOptions};

let options = RangeOptions::default();
let spread = camera_spread(&views); // once
let range = distance_range(&views, image, pixel, &sightings, t, options.tolerance_px)?;
let class = classify_range(range, spread, &options);
if class.usable() { /* compare it with the others */ }
```

## Theory

**The error along the ray.** Put the point at distance `t` along the pixel's
ray. Its error is the largest pixel distance, over the sightings in other
images, between the sighting and where the point projects in that image. At
`t = ∞` the ray is projected as a direction. An image that cannot see the point
(behind a perspective camera, or outside the lens model) makes the error
infinite. With no other sighting the error is zero everywhere.

**The tolerance.** The range is where the error stays within `tolerance_px`,
1 px by default. Sightings from affine predictions or a coarse sweep may only
meet within a pixel or two at their best distance, and would then have no
range at all, so the tolerance is widened to half a pixel more than the error
at the given distance when that is larger. An infinite error there makes the
tolerance infinite, and the range `[0, ∞]`.

**The search.** The error is not monotone in general, so the range is the
connected stretch around the given distance, found by stepping out from it.
Toward the camera the distance is halved until the error at half the current
distance leaves the tolerance, then the end is bisected ten times in log
distance between the two, keeping the side within the tolerance; the near end
is `0` when 40 halvings find no such distance. Away from the camera the same
with doubling, except that the far end is infinite when the error at infinity
is within the tolerance, or when the point is at infinity. A point at infinity
starts the near search at `1e7`. Ten bisection steps place each end within a
factor of `2^(1/1024)`, about 0.07 %, inside the true end.

**Bounded and far.** A range is bounded when its near end is above zero, its
far end finite, and far over near is at most `max_span`, 3: the photographs
say how far the point is to within a factor of three. It is far when it has no
far end and its near end is at least `far_spread`, 5, camera spreads out: the
photographs cannot tell the point from infinity, and it is not so near that
the missing far end means the reading is poor. A range is usable when it is
either; an unusable one, such as `[0, ∞]` or a range spanning a factor of ten,
supports nothing.

**The camera spread** is the largest distance between two camera centres of
the reconstruction. It is the reconstruction's own length scale, so the far
test holds whatever the units; the far-field sweep uses it for the same
reason, to put its probe point a thousand spreads out.

## Implementation notes

**The projection is the harness camera's.** A place is projected with
`ViewCamera::project_homogeneous` at `w = 1` and infinity with
`project_direction`, the rule under which a wide-angle lens (fisheye,
equirectangular) sees past 90 degrees off its axis. The track-at-pixel members
keep the stricter `project`, which refuses anything behind the image plane.

**Parity with the harness.** The search does the same arithmetic in the same
order as the Python. Over every anchor of both ground truths, full and empty
passes, every source run (146 137 anchors in 10 360 queries), the two return
the same ranges to the bit, and the harness's scores are identical. A query's
ranges take 0.1 to 0.4 ms (median) against 18 to 27 ms in Python. The harness
selects the implementation with `range_impl` (`"rust"` by default, `"python"`
for the reference).

## Python bindings

```python
from sfmtool._sfmtool import bench

near, far = bench.distance_range(edited, images, image, (x, y),
                                 [(i, (u, v)), ...], t, tolerance_px=1.0)
spread = bench.camera_spread(edited, images)
bench.classify_range((near, far), spread, max_span=3.0, far_spread=5.0)
# {"bounded": ..., "far": ..., "usable": ...}
```

`images` is a list of decoded images or an `ImagePyramidSet`, as every bench
step takes them. An image that is not one of the reconstruction's is a
`ValueError`.

## Testing

[`bench/nearby/tests.rs`](../../../crates/sfmtool-core/src/bench/nearby/tests.rs)
decides the range on the bench's synthetic capture: a point four units out seen
from cameras a metre apart has a bounded range around its distance whose ends
are where the worst sighting reaches 1 px; two cameras a centimetre apart give
a far range with no far end; a point at infinity has no far end and a near
end where the widest image moves it 1 px; a sighting 3 px off widens the
tolerance to 3.5 px; a camera with the point behind it makes the range
`[0, ∞]`; sightings in the queried image are not checked; the camera spread
and the two classes on small cases.
[`tests/rust_bindings/test_distance_range_rust_bindings.py`](../../../tests/rust_bindings/test_distance_range_rust_bindings.py)
checks the bindings on the seoul_bull fixture's longest track. Parity with the
harness's Python is measured by running the harness with `range_impl=rust`
and `range_impl=python`.
