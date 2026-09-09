# Creating a 3D point from a pixel

A reconstruction's points are what a matcher and a triangulation agreed on. A
photograph that plainly shows something the point cloud does not hold is a gap
the eye sees and the pipeline did not close, and closing it by hand means
naming the photograph and the pixel. This spec describes the function that turns
"there is something here" into a point the rest of the toolkit reads like any
other: a bearing along the pixel's own ray, with one observation in that image,
a patch frame at the size the caller names, and a patch bitmap cut from the
photograph through that frame.

The point is created **at infinity** (`w = 0`), because one sighting fixes a
direction and no distance. Two rays are what fix a place, so the second sighting
is what makes it finite: adding an observation to it triangulates it and rescales
its frame at the depth that solve found
([`add-observation.md`](add-observation.md)).

The function is pure. It takes the reconstruction value, the image, the pixel,
the patch radius and the decoded views, and returns a new value plus a report;
the base behind the input's shared pointer is never written, and the caller's
value is left exactly as it was whether the edit succeeds or is refused.

It is defined on `embedded_patches` reconstructions only, for the reason
`add_observation` is: an observation placed at a clicked pixel has a keypoint
and a patch behind it and no `.sift` feature index
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)).

Related specs:
[`edited-reconstruction.md`](edited-reconstruction.md) (the value, `PointRecord`,
`add_point`, and why an addition takes an index at or past the base's point
count), [`add-observation.md`](add-observation.md) (the edit that makes the
point finite), [`../patch/patch-cloud.md`](../patch/patch-cloud.md) (the patch
frame, and the tangent frame a `w = 0` point anchors one with), and
[`../../cli/reconstruction/xform/find-points-at-infinity.md`](../../cli/reconstruction/xform/find-points-at-infinity.md)
(the pass that moves points across the finite/infinity boundary, whose frame
rescale this pair of edits matches).

## Rust API

The function lives in
[create_point.rs](../../../crates/sfmtool-core/src/reconstruction/create_point.rs),
re-exported as
`sfmtool_core::{create_point, CreatePointOptions, CreatePointReport,
CreatePointError}`.

```rust
pub fn create_point(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f32; 2],
    radius_px: f32,
    views: &[ProjectedImage<'_>],
    options: &CreatePointOptions,
) -> Result<(EditedReconstruction, CreatePointReport), CreatePointError>;

pub struct CreatePointOptions {
    /// Written to `observation_confidence`, when the base carries it.
    pub observation_confidence: u8,
    /// Written to `normal_confidence`, when the base carries it.
    pub normal_confidence: u8,
}

pub struct CreatePointReport {
    pub point: u32,
    pub image: u32,
    pub pixel: [f32; 2],
    pub direction: [f64; 3],
    pub radius_px: f32,
    pub half_extent: f64,
    pub color: [u8; 3],
}

pub enum CreatePointError {
    NotEmbeddedPatches,
    ImageOutOfRange { image: u32, image_count: usize },
    PixelOutsideImage { pixel: [f32; 2], size: (u32, u32) },
    Unprojectable { pixel: [f32; 2] },
    BadRadius(f32),
    ViewsMissing { got: usize, expected: usize },
    Edit(EditError),
}
```

### Why it is shaped this way

**The radius is an argument, not a default.** A click names a place and says
nothing about size, and the patch's size is what every later photometric read of
this point is taken through: too small and it registers on noise, too large and
it spans a depth discontinuity. There is no value the function could pick that
would be right for both a brick and a rooftop, so the caller names one and the
function converts it. It is in **this image's pixels at this pixel**, which is
the only unit the person pointing has.

**A value in, a value out**, for the reason
[`add-observation.md`](add-observation.md) gives: the edit is a function from one
reconstruction value to the next, so a caller keeps both and undoes by pointing
at the one it had, and the returned value shares the input's base `Arc`.

**The decoded views are a named input.** A reconstruction carries poses, lenses
and thumbnails rather than photographs, and the colour and the patch bitmap come
out of a photograph. `views` is one
[`ProjectedImage`](../../../crates/sfmtool-core/src/patch/normal_refine/params.rs)
per image of the base, indexed by image index, exactly as every patch kernel
takes them; only the clicked image's entry is read.

**The report names the bearing and the frame it built**, because those are what
a caller has to be able to check without materialising: `direction` is the unit
ray, `half_extent` is what `radius_px` became.

### Example

```rust
use sfmtool_core::{create_point, CreatePointOptions, EditedReconstruction};

let edited = EditedReconstruction::new(base);
let (next, report) = create_point(
    &edited,
    7,                   // the image the user pointed in
    [120.5, 88.25],      // where they pointed
    12.0,                // the patch radius, in that image's pixels
    &views,              // one ProjectedImage per image
    &CreatePointOptions::default(),
)?;

assert_eq!(next.point(report.point).expect("created").point().w, 0.0);
assert!(std::sync::Arc::ptr_eq(&edited.base, &next.base));
```

## What the call does

### 1. The refusals

The call refuses, producing nothing and touching nothing, when the base's
observations are `.sift` feature indexes, when the image index is past the image
table, when fewer views were supplied than the base has images, when the radius
is not a positive finite pixel count, when the pixel is outside the image's own
sensor rectangle, and when the camera model has no ray for the pixel or for the
pixel one radius away from it. The order is the cheap checks first, so a caller
greying a menu entry gets the same answers without decoding anything.

### 2. The bearing

The point's coordinate is the clicked pixel's own ray: `pixel_to_ray` through
the image's camera model, distortion included, rotated into world by the
camera-to-world rotation and normalised. `w` is `0`, which is what says the
coordinate is a direction rather than a place.

### 3. The frame

`radius_px` becomes an angle rather than a length, because a bearing has no
depth for a length to be measured at. The angle is measured through the camera:
the pixel one radius away along each sensor axis is unprojected too, and the
angle between each of those rays and the centre ray is the half-angle that axis
subtends. Doing it this way rather than as `radius / focal` puts the lens's own
distortion in the answer, which at a wide-angle frame edge is a large
correction. A pixel a radius away would fall off the sensor at the frame edge,
so the probe steps whichever way stays on it; the angle is the same either way
to the accuracy this is measured at.

The two half-angles are averaged into one, because **a patch frame is square**:
its two half-vectors have equal length and its bitmap is `R x R`. The stored
half-vector length is `tan` of that half-angle, which is what makes the frame
subtend it: an infinity patch's corner is the direction `d + s·u`, and with
`u ⊥ d` and `|d| = 1` that corner sits at `atan(|u|)` off `d`.

The frame itself is the tangent-sphere frame the format states for a `w = 0`
point ([`../patch/patch-cloud.md`](../patch/patch-cloud.md)): `u` and `v`
perpendicular to `d` with `u × v` along `−d`. Its rotation about `d` is pinned
by the camera's own up axis, so the stored patch is the upright crop the person
pointing is looking at rather than an arbitrary rotation of it.

### 4. The bitmap and the colour

When the base carries `patch_bitmaps_y_x_rgba`, the created point gets one at
that column's own resolution: the frame rendered into the clicked image by
`WarpMap::from_patch` and the mip-correct bilinear sampler, which is the pair
every stored patch bitmap is produced by. A pixel the warp cannot sample stays
black, and the alpha channel is opaque everywhere, because the whole tile is
content this one image saw. A base that carries no bitmap column gets a record
with no bitmap; a record carries exactly the base's columns and nothing else.

The point's colour is the image sampled bilinearly at the pixel. A single-channel
image gives the same value in all three channels.

### 5. The columns the click says nothing about

- **The normal** is zero, which is what a `w = 0` row carries: a direction has no
  surface orientation, and the demotion pass zeroes it for the same reason.
- **`normal_confidence`**, when the column exists, is zero with it. The format
  keeps the two coherent for a `w = 0` row.
- **`observation_confidence`**, when the column exists, is the column's maximum.
  It is a photometric agreement score, and there is no second view for this
  sighting to agree with: the person pointing is the evidence, and the column's
  own maximum is the honest reading of a sighting placed by hand rather than a
  fabricated correlation.
- **The constraint**, when the columns exist, is free: a created point states
  nothing about a distance a caller owns.
- **The stored error** is zero. No reprojection has been measured.

The record is produced through `add_point`, so it takes an index at or past the
base's point count, `replaces` is `None`, and a materialisation appends it.
`RowMap::by_scan` over the materialised value reports the row as created, which
is what a caller following an index across a bulk edit needs it to say.

## Implementation notes

**The bearing is unit-length, and stays so.** Everything downstream reads a
`w = 0` coordinate as a unit direction -- the renderer, the tangent frame, the
promotion in [`add-observation.md`](add-observation.md) -- so the normalisation
happens once, here, rather than being re-asserted at each reader.

**The probe direction is chosen per axis, not per call.** A click in the last
`radius_px` columns of the sensor has no pixel a radius to its right, and a
click in the first has none to its left; each axis steps toward whichever side
is on the sensor, and a radius wider than the sensor itself is what
`BadRadius` then refuses.

## Testing

`crates/sfmtool-core/src/reconstruction/create_point/tests.rs` builds on the
scene [`add-observation.md`](add-observation.md)'s tests build -- pinhole
cameras looking down world `+z` at a textured plane -- so a click at a known
projection has a known ray behind it. What it pins:

- Each refusal by its own variant: a `sift_files` base, an image past the table,
  a pixel off the sensor on each side, a radius of zero, a negative one and a
  `NaN` one, and too few views.
- The stored bearing is the clicked pixel's own ray, to a milliradian, and it is
  a unit; the one observation is in the clicked image at the clicked pixel.
- `radius_px` at the principal point of a pinhole gives a half-extent of exactly
  `radius / f`, for three radii; the two half-vectors are equal in length and
  perpendicular to the bearing.
- Every optional column is filled the way the schema needs: an `R x R x 4`
  bitmap that is opaque and not blank, the confidence column's maximum, a zero
  normal confidence and a zero normal.
- The returned value shares the input's base by pointer and the input is
  unchanged.
- A materialisation appends the point, and `RowMap::by_scan` over the base and
  the materialised value reports its row as created and carries the base's own
  point over.
- The point-edit hash of the created record is the same for two identical clicks
  and different for a different pixel.
- Creating a point and then adding a second observation to it, clicked a couple
  of pixels off the truth: the two-pass fit reports a finite ZNCC above 0.9,
  walks the sighting off the click and toward the true projection, `w` crosses
  from 0 to 1, the point lands within a twentieth of a unit of the truth and the
  frame is scaled at the placement distance. A second sighting along the same
  bearing is refused by the provisional triangulation.

## Non-goals

- Choosing the radius. Nothing in the reconstruction says how large the thing
  under the pixel is; the caller decides, and
  [`../../gui/edits/create-point.md`](../../gui/edits/create-point.md) is where
  the viewer's data-derived offer is described.
- Placing the point at a finite depth from one view. A single ray does not have
  one, and a guessed depth would be a claim the data does not make.
- Refining the clicked pixel photometrically. There is no second view to
  register against, and the patch being registered would be the one this call is
  cutting from that same pixel.
- Creating a point on a `sift_files` reconstruction, which would need the format
  to carry an observation with no feature behind it.
