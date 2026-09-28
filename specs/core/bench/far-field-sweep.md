# The far-field sweep

A pixel of a photograph can show something so far away that the photographs,
taken a few metres apart, barely see it move: a skyline, a tower across a bay,
the sky. Matching keypoints and triangulating says little about such a pixel,
because the rays meet at a shallow angle or not at all. The far-field sweep
asks the photographs directly. It takes the small patch of the photograph
around the pixel, places it at a series of distances along the pixel's ray from
infinity inwards, and reads how well every other photograph agrees with it at
each distance. Each distance where the agreement peaks is returned as a
reading: a point (or a direction, at infinity) with the range of distances it
allows, the images that agree, and the numbers the reading rests on. Nothing is
written to the reconstruction.

The sweep is the far test of the anchor finder in the track-at-pixel harness
([`scripts/track_at_pixel/anchors.py`](../../../scripts/track_at_pixel/anchors.py),
`from_farfield`), where it was designed and measured, and the first part of
that finder moved into core (combined in [nearby-tracks.md](nearby-tracks.md)). Its
readings are hypotheses, not a decision: near a pixel the scene can hold a far
surface and a nearer one in front of it, and which of them the pixel shows is
for a comparison between readings from several sources to settle.

It is built from two smaller pieces that are public because later steps of the
finder use them too: **reading a patch along a ray**, and **projecting a
direction**, the camera's `w = 0` projection.

## Rust API

The pieces live in
[`bench/nearby/`](../../../crates/sfmtool-core/src/bench/nearby.rs): the sweep
in [`far_field.rs`](../../../crates/sfmtool-core/src/bench/nearby/far_field.rs),
the patch read in
[`patch_read.rs`](../../../crates/sfmtool-core/src/bench/nearby/patch_read.rs)
and the grey images it samples in
[`grey.rs`](../../../crates/sfmtool-core/src/bench/nearby/grey.rs), all
re-exported from `sfmtool_core::bench`. The direction projection is
`ViewCamera::project_direction` in
[`track_at_pixel/neighbourhood.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel/neighbourhood.rs),
crate-internal, shared with the track-at-pixel members. The sweep is bound as
`sfmtool._sfmtool.bench.far_field_sweep` and the patch read, in
[`patch_read.rs`](../../../crates/sfmtool-py/src/bench/patch_read.rs), as
`sfmtool._sfmtool.bench.read_patch_along_ray`.

```rust
pub fn far_field_sweep(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    image: u32,
    pixel: [f64; 2],
    options: &FarFieldOptions,
    progress: &Progress<'_>,
) -> Result<FarFieldSweep, FarFieldError>;

pub struct FarFieldSweep {
    pub readings: Vec<FarFieldReading>, // kept, highest peak first
    pub dropped: Vec<FarFieldReading>,  // peaks the refit dropped, with why
}

pub struct FarFieldReading {
    pub disparity: f64,              // the sweep's, in px of the widest image
    pub position: Vector3<f64>,      // a place, or a unit direction
    pub at_infinity: bool,
    pub views: Vec<(u32, [f64; 2])>, // the queried image first
    pub query_pixel: [f64; 2],       // the pixel, or where a moved reading lands
    pub distance_px: f64,
    pub max_reproj_px: f64,
    pub max_ray_angle_deg: f64,
    pub depth: f64,                  // along the queried camera's axis
    pub range: Option<[f64; 2]>,     // None once moved
    pub metrics: FarFieldMetrics,
    pub grouping: Option<FarFieldGrouping>,
    pub refit: Option<Refit>,        // Agrees | Grouped | Unsplit | Stands | Moved | ...
    pub refit_px: Option<f64>,
}

pub fn read_patch_along_ray(
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    patch: &RayPatch,        // image, pixel, radius_px
    distances: &[f64],       // f64::INFINITY allowed
    images: &[u32],
    keep_samples: bool,
) -> Option<PatchRead>;      // None: the queried patch is flat or off its photograph

pub struct PatchRead {
    pub images: Vec<u32>,
    pub distances: Vec<f64>,
    pub whole: Array2<f64>,   // [distance, image] ZNCC, -1 where unread
    pub middle: Array2<f64>,  // the same samples' middle 5 x 5
    pub centres: Array3<f64>, // where the pixel lands in each read
    pub middle_std: f64,      // the queried middle's grey standard deviation
    pub samples: Option<PatchSamples>,
}
```

**Why the grey images are an argument.** Every read samples the photographs in
grey, blurred by a Gaussian of one pixel, and converting a photograph costs far
more than a query reads of it. `GreyImages` is a cache of one lazily built grey
image per view, filled through a shared reference, which the caller builds once
per set of views and passes to every query, the same way the decoded views are
the caller's. The Python binding keeps one inside each `ImagePyramidSet`.

**Why the result keeps the dropped peaks.** The harness returns only the kept
readings, and so does the binding. A dropped peak still says something a
caller may show: that the photographs agreed on a distance until the grouping
split them and the refit landed off the image or too far away.

**Why one reading type for moved and unmoved readings.** A reading the refit
moves is still the sweep's peak, sighted by a different group of images; the
`refit` outcome says which, and the one field whose meaning changes, `range`,
is absent once moved, because the moved point's range is its sightings' to
compute.

```rust
use sfmtool_core::bench::{far_field_sweep, FarFieldOptions, GreyImages};
use sfmtool_core::progress::Progress;

let grey = GreyImages::new(views.len()); // once, kept beside the views
let found = far_field_sweep(&edited, &views, &grey, image, pixel,
                            &FarFieldOptions::default(), &Progress::none())?;
for r in &found.readings {
    println!("{} px: {:?} {:?}", r.disparity, r.range, r.refit);
}
```

## Theory

**Distances as disparities.** For a point far from the cameras, the shift it
makes in another photograph is close to linear in its inverse distance. So the
sweep counts distances as disparities `d` in the image that moves the pixel
most, among those it lands in at infinity: `d = 0` is infinity and `R / d` the
distance along the pixel's ray, `R` that image's pixels per unit of inverse
distance. `R` is measured, not derived: the pixel's ray is projected as a
direction and as a point `1000` camera-spreads out, and the shift between the
two times that distance is the rate. The sweep reads 0, 1, 2, 3, 4, 6, 8, 10,
12, 14 and 16 px.

**Reading the patch.** At each distance the patch is an 11 x 11 grid of samples
8 px from the pixel to its edge, on the plane that faces the queried camera at
that distance along the pixel's ray; at infinity the grid's rays are projected
as directions. Each read gives two ZNCCs of the same samples with the queried
patch: of the whole grid and of its middle 5 x 5. When the whole matches and
the middle does not, the parts away from the pixel carry the match: a small
near object whose patch is mostly background, a depth edge, a pattern that
repeats along the epipolar line.

**Wide images and the reading.** An image close to the queried camera reads
alike at every disparity, so it cannot tell distances apart. The reading at a
disparity is over the **wide** images: those whose whole patch reads 0.8
somewhere in the sweep, and that move the pixel at least half as far as the
widest of those. Judging width among the images that match keeps an image that
sees something else at the pixel from setting the scale. The reading is the
middle's mean over the three best wide images, or the whole patch's when the
queried middle is flat (grey standard deviation under 8), since a flat middle
correlates with noise.

**Peaks, not the best.** Near a pixel the scene can hold a far surface and a
nearer one, or two lookalikes, and with one or two wide images there is nothing
in the reading itself to choose between them. So every peak is a reading, up to
three, the highest first. A peak must read 0.8 on the whole patch and, unless
the middle is flat, 0.7 on the middle; it must stand 0.02 above the reading
around it (its **prominence**: the height above the lowest value between it and
the nearest higher value on either side, the higher of the two sides, or above
the lowest value of all for the highest peak), since images that barely move
give a flat reading with no peak; and it must not be the last disparity, where
a rising reading says the peak is further in than the sweep reaches. A
reading's range runs to the midpoints with the neighbouring disparities, to
infinity for the first.

**Which images belong.** The sweep compares each image with the query only, so
one peak can gather images of two surfaces: the query's, and one that stands in
front of it from a few cameras close together and resembles the query's patch
as a whole. The query and the agreeing images, at most 16, are compared pair by
pair by the ZNCC of their middles (whole patches when the middle is flat), and
grouped by average linkage, merging while the two closest groups' mean is 0.9
or more. The reading keeps the query's group (`Agrees` when that is every
image, `Grouped` otherwise). When the query stands alone and no two other
images group either, nothing contradicts the sweep and it stands (`Unsplit`).
When two or more other images group, they agree with each other and not with
the pixel: that group is built into a bench track with the query turned out
and fitted. Where it lands within 8 px of the pixel the reading stands on that
group (`Stands`); further out, within 48 px and with every sighting within 2 px
of the fitted point, the reading moves there (`Moved`); otherwise, or where the
fit lands off the image or cannot be built, the peak is dropped.

**Measured.** In the harness, against the ground truths of seoul_bull and Kerry
Park, the sweep gives a reading at the pixel for Kerry Park's points 300 m and
more away that the earlier infinity test missed (48 of 59), and its wrong
readings mostly rank below the right depth layer. The harness's
[README](../../../scripts/track_at_pixel/README.md) (*The far-field sweep*)
records the runs and the cases each rule was added for.

## Implementation notes

**The grey images are the harness's.** The harness sampled `cv2` images, and the
sweep's thresholds were chosen on them, so the conversion is OpenCV's: grey
from RGB with its 15-bit fixed-point weights (9798, 19235, 3735, rounded, shift
15), exact; a nine-tap separable Gaussian of sigma 1 with the image reflected
at its borders without repeating the edge pixel (`BORDER_REFLECT_101`); and a
bilinear sample in floating point with no value where any of its four pixels
is off the image, as `cv2.remap` with a NaN border gives (OpenCV 5 interpolates
in floating point; earlier versions quantised coordinates to 1/32 px). The
residual difference is in the last bits of a 32-bit float, from the order the
blur and the interpolation add their terms; the ZNCCs are computed in `f64`
where the harness used `f32`. Over every far-field query of both ground truths
the two implementations return the same readings.

**Two reads per sweep.** The first read, at infinity, over every other image,
only finds the images the pixel lands in; the second reads those at every
disparity, keeping the samples for the pairwise grouping. The queried patch is
sampled again for each read, which costs 121 samples.

**The pairwise table is O(N²)** in the images of a reading, which is why it is
capped at 16, the query and the 15 best-reading others. How the grouping should
scale to a reading seen in a few dozen images is not settled.

**A flat image in the pairwise table.** An agreeing image whose middle is
constant has no direction to normalise; its row of the table is `NaN`, every
mean with it is `NaN`, and average linkage never merges it, as `numpy` did.

**The refit's track** is seeded like the harness's
`candidates/common.track_from_sightings`: a cluster at the pixel of the patch's
radius with the bench's default thresholds, each sighting added and set `in`,
the upgrade to the track stage, then the query turned out and two default fits.
It shares `seed_cluster_with` and `upgrade_sightings` with the track-at-pixel
finish.

## Parameters

`FarFieldOptions::default()` in
[`far_field.rs`](../../../crates/sfmtool-core/src/bench/nearby/far_field.rs); the
harness names in brackets.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `disparities` | `0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16` | the disparities read, in px of the widest image (`ff_disparities`) |
| `radius_px` | `8.0` | the patch's half-width in the queried image (`inf_radius_px`) |
| `wide` | `0.5` | the share of the widest image's rate an image needs to be wide (`ff_wide`) |
| `wide_among` | `Matching` | the widest among the matching images, or `All` (`ff_wide_among`) |
| `min_whole` | `0.8` | a peak's whole reading, and an agreeing image's (`ff_min_whole`) |
| `min_middle` | `0.7` | a peak's middle reading unless the middle is flat (`ff_min_middle`) |
| `middle_min_std` | `8.0` | the queried middle's grey standard deviation under which it is flat (`inf_centre_min_std`) |
| `max_peaks` | `3` | the most peaks kept (`ff_max_peaks`) |
| `min_prominence` | `0.02` | how far a peak must stand above its surroundings (`ff_min_prominence`) |
| `refit` | `true` | group the images pairwise and refit (`ff_refit`) |
| `group_cut` | `0.9` | the average-linkage cut (`ff_group_cut`) |
| `group_max` | `16` | the most images in the pairwise table (`ff_group_max`) |
| `refit_px` | `8.0` | a refit landing this near the pixel leaves the reading in place (`ff_refit_px`) |
| `refit_max_px` | `48.0` | the furthest a reading may move (`ff_refit_max_px`) |
| `refit_max_err_px` | `2.0` | the largest sighting error of a moved reading (`ff_refit_max_err_px`) |

## Python bindings

```python
from sfmtool._sfmtool import bench

readings = bench.far_field_sweep(edited, images, image, (x, y),
                                 options={"wide": 0.4, "refit": False})
```

`images` is a list of decoded images or an `ImagePyramidSet`, as every bench
step takes them; a set also keeps its grey images between calls. `options`
overrides `FarFieldOptions` fields by name (`wide_among` as `"matching"` or
`"all"`); an unknown key is a `ValueError`, as is an image or pixel that names
no place. The result is a list of the kept readings as dicts with the keys the
harness's anchors carry: `source` (`"farfield"`), `id`, `position`, `w`,
`views` (`[image, x, y]` rows), `query_pixel`, `distance_px`, `n_views`,
`max_reproj_px`, `max_ray_angle_deg`, `depth`, `farfield` (the metrics, with
`group_middle` and `left_out` from the grouping), and with the refit `groups`,
`query_middle`, `refit` and `refit_px`. A reading at the sweep's distance
carries `disparity` and `range_override`; a moved one `sweep_disparity`
instead. The harness's `from_farfield` calls it by default (`ff_impl="rust"`).

```python
read = bench.read_patch_along_ray(edited, images, image, (x, y), 8.0,
                                  [math.inf, 40.0, 20.0], read_images=None,
                                  samples=False)
```

`read_patch_along_ray` reads in every image but the queried one when
`read_images` is left out, and returns `None` for a flat or off-photograph
patch, or a dict with `images`, `distances`, `whole` and `middle`
(`(distances, images)` arrays, `-1` where unread), `centres`, `middle_std`
and, with `samples`, `template`, `values` and `middle_mask`.

## Testing

[`bench/nearby/tests.rs`](../../../crates/sfmtool-core/src/bench/nearby/tests.rs)
decides the pieces on the bench's synthetic capture: a plane put so far out
that it reads at disparity 0, as a bearing, with every image in one group; a
plane at the distance of disparity 6 reading there and nowhere at infinity; a
plane too near for the sweep giving no far reading; the patch reading best at
the plane's distance and landing where the scene projects; a flat or
off-photograph patch not read; the grey conversion and blur against OpenCV's
numbers; best-three, prominence and average linkage on small cases.
[`tests/rust_bindings/test_far_field_sweep_rust_bindings.py`](../../../tests/rust_bindings/test_far_field_sweep_rust_bindings.py)
checks the binding's keys, that an image list and a pyramid set read alike,
the overrides and the refusals. Parity with the harness's Python
implementation is measured by running the harness with `ff_impl=rust` and
`ff_impl=python`.

## Non-goals

The sweep does not decide which reading is the pixel's; the depth layers of the
[nearby-tracks finder](nearby-tracks.md) do. It does not read nearer than its last disparity; the
matching sources cover near geometry.

## Open questions

How the pairwise grouping should scale past its cap of 16 images, and whether
the best-reading 16 are the right ones to keep when a reading has more.
