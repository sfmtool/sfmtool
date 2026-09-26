# Bundle-adjusting a reconstruction

A reconstruction is a set of camera poses and a set of 3D points, and the only
evidence for either is where the points landed in the photographs. Bundle
adjustment is the joint refinement that takes all of that evidence at once: it
moves every pose and every point together until the pixels they predict sit as
close as they can to the pixels that were actually measured. This spec describes
the function that runs one over a whole reconstruction value: what it gathers,
what it refuses, what it writes back, and what it does with a point the solve can
no longer support.

It is not the solver. The solver is the staged robust array kernel in
[`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md), which
takes poses, points and observations as plain arrays over a list of cameras.
What this adds is the layer between that kernel and a reconstruction: which
images are posed, which cameras they were taken through, which observations
carry a pixel, what each point's stored constraint means, and how the answer goes back into a value whose patch frames,
colours and tracks have to come out the other side still describing the same
scene.

The function is pure. It takes the reconstruction value and its options, and
returns a new value plus a report; the input is left exactly as it was whether
the adjustment succeeds or is refused.

Related specs:
[`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md) (the
kernel, its schedule, its point constraints and its points at infinity),
[`edited-reconstruction.md`](edited-reconstruction.md) (the value, and the row
map a caller follows an index through),
[`../bench/editable-track.md`](../bench/editable-track.md) (the bench fit, which
rescales a patch frame the same way when a track crosses between a place and a
bearing), and
[`../../gui/edits/bundle-adjust.md`](../../gui/edits/bundle-adjust.md) (the
viewer's edit over it).

## Rust API

The function lives in
[bundle_adjust.rs](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust.rs),
re-exported as `sfmtool_core::{bundle_adjust, BundleAdjustOptions,
BundleAdjustReport, CameraAdjustment, BundleAdjustError}` and bound as
`EditedReconstruction.bundle_adjust`. `CameraRelease` is the kernel's type,
[`../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs`](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs),
re-exported from this module.

```rust
pub fn bundle_adjust(
    recon: &SfmrReconstruction,
    options: &BundleAdjustOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, BundleAdjustReport), BundleAdjustError>;

pub struct BundleAdjustOptions {
    /// One per camera in the camera table, or empty for every camera held.
    pub releases: Vec<CameraRelease>,
    pub schedule: Vec<BaSchedule>,
    pub max_iters: usize,
    pub min_track: usize,
    pub min_obs: usize,
}

impl BundleAdjustOptions {
    /// The defaults with `release` for each of `camera_count` cameras.
    pub fn uniform(camera_count: usize, release: CameraRelease) -> Self;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraRelease {
    pub focal: bool,
    pub distortion: bool, // k1, or the radial spline
}

impl CameraRelease {
    pub const HELD: CameraRelease;                 // neither
    pub const FOCAL: CameraRelease;                // the focal alone
    pub const FOCAL_AND_DISTORTION: CameraRelease; // both
}

pub struct BundleAdjustReport {
    pub images: usize,
    pub points: usize,
    pub observations: usize,
    pub points_deleted: usize,
    pub median_residual_before: f64,
    pub median_residual_after: f64,
    pub cameras: Vec<CameraAdjustment>, // one per camera in the solve
}

pub struct CameraAdjustment {
    pub camera: usize,        // its index in the reconstruction's camera table
    pub images: usize,        // posed images taken through it in the solve
    pub focal_before: f64,
    pub focal_after: f64,
    pub focal_released: bool,      // what its release asked for, and so what moved
    pub distortion_released: bool,
    pub outermost_observed: Option<KeypointReach>, // under the solved camera
}

pub enum BundleAdjustError {
    NoKeypoints,
    ReleaseCount { releases: usize, cameras: usize },
    FocalNotReleasable { camera: usize, camera_model: &'static str },
    DistortionWithoutFocal { camera: usize },
    DistortionNotReleasable { camera: usize, camera_model: &'static str },
    NoPosedImages,
    NoObservations,
    EmptySchedule,
    Constraints(PointConstraintsError),
    Cancelled,
    Degenerate { observations: usize, min_obs: usize },
}

/// Whether the kernel's analytic focal column is exact for this camera.
pub fn focal_is_releasable(camera: &CameraIntrinsics) -> bool;

/// Whether this camera has lens distortion the adjustment can release.
pub fn distortion_is_releasable(camera: &CameraIntrinsics) -> bool;
```

### Why it is shaped this way

**A value in, a value out.** The adjustment is a function from one
reconstruction to the next, which is what lets a caller keep both and undo by
pointing at the one it had ([`edited-reconstruction.md`](edited-reconstruction.md)).
There is no `&mut` on the input, so a refusal cannot leave a half-adjusted
reconstruction behind, and the caller decides whether the answer replaces
anything.

**One options struct, and a small one.** The kernel takes eighteen arguments
because it is an array kernel with every switch exposed. A caller here is
adjusting a reconstruction, and the only decision that is genuinely theirs is
which lenses move; everything else is the schedule the rest of the toolkit
runs. So `releases` is the first field and the rest are the kernel's own
defaults, stated rather than hidden so a caller that does need to tighten a trim
can.

**The releases are per camera.** Each camera has its own model, and a model
decides what can be released at all, so the choice is made camera by camera: a
rig that mixes an `OPENCV_FISHEYE` camera with spline cameras releases the
spline cameras and holds the other. `releases` holds one `CameraRelease` per
camera-table entry, in table order, because the table index is the name every
other surface gives a camera; an entry for a camera no posed image uses is
ignored, since that camera is not in the solve. An empty list holds every
camera, which is what `Default` gives, and any other length than the table's is
refused (`ReleaseCount`) rather than read as a prefix, because a list one entry
short would otherwise hold the last camera without saying so.
`BundleAdjustOptions::uniform(n, release)` builds the case where every camera
is released alike, as `BaCameras::shared` builds the one-camera case of the
kernel's camera list.

**A camera with nothing released is held, and a solve with every camera held
still runs.** Holding a lens is a choice, not an error: the solve refines the
poses and the points against the lenses as they are, and the report's entries
say `focal_released: false` and `distortion_released: false`. That is also the
default, because a lens that moves is a different claim about the capture than
a pose that does, and the caller should be the one making it.

**A release a model cannot take is refused, naming the camera.** The kernel
holds a parameter its model is not exact for and releases the rest. That is
right for a kernel with a parity requirement and wrong for a caller who asked a
question: a report saying the focal was released when it was held would be
false. So each camera's release is checked against its own model, and the first
camera that fails refuses the call with its table index and its model:
`FocalNotReleasable` for a focal release on a model the kernel's analytic focal
column is not exact for, and `DistortionNotReleasable` for a distortion release
on a model with no distortion the kernel can free.
[`focal_is_releasable`] and [`distortion_is_releasable`] are public and per
camera for the same reason, so a caller offering the releases as choices can
grey each choice on its own camera instead of taking it and refusing.

**The distortion is released only with the focal, per camera.** The kernel has
two distortion rungs, each exact on its own models: `opt_k1` frees `k1` on
`SIMPLE_RADIAL_FISHEYE`, and `opt_bspline` frees the spline on `SFMTOOL_FISHEYE`
and `SFMTOOL_PINHOLE`. No model carries both, so a caller here asks one
question per camera, whether its lens distortion moves, and
`CameraRelease::distortion` reaches both rungs. Neither `k1` nor the spline can
change the scale at the centre of the image, which is the focal's job:
`θ·(1 + k1·θ²)` has slope one on the axis, and the spline's gauge pins its value
and slope there. A distortion released against a held focal could only bend the
periphery around a scale it cannot fix, so a camera whose release has the
distortion without the focal is refused (`DistortionWithoutFocal`, naming it).

**The kernel is handed the choice, not re-told it.** The kernel's three flags
are requests to every camera; `BaCameras::releases` narrows them per camera
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)). This
function passes every flag on and each camera's checked release in that list,
so the kernel frees exactly what the report says it did.

**The coefficient count and the domain are not the adjustment's.** How many
coefficients a spline has, and where its domain ends, choose how the lens curve
is parameterized; the solve refines the coefficients it is given and cannot
change either. Changing them is a refit of one camera, a switch of that camera
to its own spline model
([`switch-camera-model.md`](switch-camera-model.md)), after which an
adjustment with its distortion released brings the new coefficients to the
observations.

**The report carries each camera's outermost observation.** Measured under the
camera the solve returned, as a radius and an incidence angle, because that is
the camera whose domain the next adjustment would edit. Only the observations:
the adjustment reads nothing off disk, so the features detected in the images'
`.sift` files are for a caller to read with `outermost_keypoints`.

**The report is per camera.** Each camera in the solve has its own focal, so
the report carries one `CameraAdjustment` per camera rather than one focal for
the whole solve. Per-camera residual medians are not in it: the
overall medians are the ones a caller reports in one line, and a caller wanting
more runs the kernel.

**The report names three populations.** `images`, `points` and `observations`
are what went into the solve, which is not the whole reconstruction: an unposed
image is not in it, and neither is a point nothing posed observes.
`points_deleted` is what came out short. A caller reporting the edit in one line
needs all four to write a true sentence.

**The errors are an enum.** Every variant is a property of the value or the
options that a caller could have checked, and two of them (`NoKeypoints`,
`NoPosedImages`) are exactly what a menu entry greys itself on. A
`Result<_, String>` would make that gate a string comparison.

### Example

```rust
use sfmtool_core::progress::Progress;
use sfmtool_core::{bundle_adjust, BundleAdjustOptions};

use sfmtool_core::reconstruction::bundle_adjust::CameraRelease;

// A two-camera rig: release camera 1's lens and hold camera 0's.
let options = BundleAdjustOptions {
    releases: vec![CameraRelease::HELD, CameraRelease::FOCAL_AND_DISTORTION],
    ..BundleAdjustOptions::default()
};
let (next, report) = bundle_adjust(&recon, &options, &Progress::none())?;
println!(
    "{} px -> {} px, {} points dropped",
    report.median_residual_before, report.median_residual_after, report.points_deleted,
);
for camera in &report.cameras {
    println!(
        "camera {}: focal {} -> {} over {} images",
        camera.camera, camera.focal_before, camera.focal_after, camera.images
    );
}
```

## What the call does

### 1. The refusals

In order, cheapest first, so a caller greying a menu entry gets the same answers
without solving anything:

- An empty schedule (`EmptySchedule`).
- Observations with no pixel behind them (`NoKeypoints`): a `sift_files` value
  without the format's optional inline keypoint column states no coordinate for
  a sighting, and the residual has nothing to be measured against. The other
  observation mode always carries one.
- No image with a finite pose (`NoPosedImages`).
- A release list whose length is neither zero nor the camera table's
  (`ReleaseCount`).
- Then, for each camera the posed images use, in table order, the first of: a
  distortion release without the focal (`DistortionWithoutFocal`); a focal
  release on a model the kernel's focal column is not exact for
  (`FocalNotReleasable`); and a distortion release on a camera that is neither a
  `SIMPLE_RADIAL_FISHEYE` nor a spline model whose spline is defined, at least
  two coefficients on a positive finite domain end (`DistortionNotReleasable`).
  Each names the camera by its table index, and the last two its model.
- No observation of any point in a posed image (`NoObservations`).
- Constraint columns stating something the adjustment cannot honour
  (`Constraints`), by the rules of
  [`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md).

The last refusal is the solve's own, in step 5.

### 2. What goes into the solve

Every image with a finite pose, in table order, and every observation whose image
is one of them. An unposed image is not a camera a residual can be written
against, so it sits the solve out and its observations do not enter; a point left
with no observation in the solve is not in the solve either, and comes back
exactly as it went in.

The cameras of the solve are the camera-table entries the posed images use, in
table order, and each posed image's index into that list goes to the kernel as
its image-to-camera column. The posed images may use any number of cameras; each
keeps its own lens in the solve. A camera only unposed images use, or none, is
not in the solve and comes back exactly as it went in.

Every live point goes in at the coordinate it holds, with the representation it
holds: a `w = 0` row is a world-frame direction and enters the kernel's
`point_at_infinity` mask. That mask is honoured for the **whole** solve. No point
crosses between a bearing and a position here, because a crossing is a claim
about how much of the geometry a track can state, and deciding that is
[`../../cli/reconstruction/xform/find-points-at-infinity.md`](../../cli/reconstruction/xform/find-points-at-infinity.md)'s job
rather than a side effect of a refinement.

A point's constraint is the one its constraint columns state, converted through
`PointConstraints::from_arrays`, which is the same bridge the Python binding's
own constraint arguments go through. A reference image is re-indexed onto the
solve's image list first, and a point whose reference is not in it comes back
free rather than carrying a distance from a camera the solve does not hold, which
is the rule
[`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md) states
for a consumer that drops an image.

### 3. The two runs

The kernel runs twice. The second is the adjustment. The **first** runs it over
an **empty** schedule, which executes no round and reports the residuals at the
state it was handed: the "before" median is then measured by the same projection
the "after" one is, through the same camera model, rather than by a second
spelling of the reprojection in this module. Both runs start from the cameras
the value holds, so the "before" median describes the input value.

Both medians are taken over the finite residuals of the points that survive, so
the two numbers describe one population and their difference is the improvement
on it.

### 4. Writing the answer back

- **Poses.** Each posed image takes the rotation and translation the solve
  ended with.
- **The cameras.** Each camera in the solve is replaced by the camera the
  kernel returned for it: itself at its solved focal where its release has the
  focal, with its solved `k1` or spline where its release has the distortion,
  and itself unchanged where it was held. Nothing else about any lens moves: the principal
  point, and every other model's distortion, stay where they are.
- **Positions and representations.** Each point in the solve takes its solved
  coordinate, and `w` follows the kernel's returned representation.
- **The stored error.** Each point's error column becomes the RMS of that
  point's own finite residuals at the final state. The column describes how well
  the point is reconciled with its observations, and after a solve that moved
  everything the stored value would otherwise describe a geometry the value no
  longer holds.
- **Patch frames.** See "The frame follows the depth" below.
- **Everything else per point** -- the bitmap, the normal, the colour, the
  constraint, the track and every keypoint in it -- is the point's own and is
  kept. The adjustment moves geometry; it does not re-fit appearance and it moves
  no observation.

The derived indexes are rebuilt, so the observation offsets, the feature maps and
the infinity-point count describe the value that comes back.

### 5. The points that do not come back

A point in the solve is **deleted** when its coordinate came back non-finite, or
when every one of its observations is invalid at the final state. Both are the
same statement: nothing that saw the point can still be reconciled with it. A
track the trim wears below `min_track` arrives here by the first route, because
the kernel's inter-round re-estimation cannot rebuild a track with fewer than two
usable rays and returns it non-finite.

Deleting is the honest outcome. A point kept at a coordinate no observation
supports is a claim about the scene with nothing behind it, and one kept at a
non-finite coordinate is not a claim at all. The points that survive keep their
order, so the caller reads what happened off
`RowMap::by_scan` ([`edited-reconstruction.md`](edited-reconstruction.md)) over
the input and the output.

Every residual coming back infinite is not a per-point verdict but the kernel's
**degenerate exit**: fewer than `min_obs` observations survived a trim, and the
state passed through untouched. Read as a verdict it would delete the whole point
set, so it is the call's refusal (`Degenerate`) instead, and no value comes back.

### The frame follows the depth

A point's patch frame is two half-vectors: world-space extents on a finite point,
and angular extents tangent to the direction sphere on a bearing. The distance
that converts between them is the **placement distance**, the distance from the
camera-cloud centroid, which is where
`SfmrReconstruction::materialize_points_at_infinity` places a bearing and what a
bench fit multiplies by and divides by when a track crosses
([`../bench/editable-track.md`](../bench/editable-track.md)). Here it is
`ImageTable::placement_scale`, measured from the same reference so the
conversions cancel exactly.

An adjustment moves points in depth without changing what they look like, so a
frame left alone would change the patch's apparent size in every view that sees
it. Each point's frame is therefore multiplied by the ratio of its placement
distances, after over before, measured against the poses each state holds. A
point that stayed where it was scales by one; a point that halved its distance
from the centroid halves its frame; a degenerate distance (a point at the
centroid itself, which subtends no angle) leaves the frame alone rather than
destroying it.

## Implementation notes

**The "before" residuals come from the kernel, not from a second projection.**
Running the kernel over an empty schedule is the whole of it: the staged loop
iterates the schedule, so with no rounds it falls straight through to the final
residual pass. That pass is model-aware in ways a re-implementation here would
have to copy exactly -- the ray-path in-front measure, the invalid-observation
sentinel, the direction rows projecting through rotation alone -- and a copy
would drift.

**Points not in the solve are not candidates for deletion.** The deletion rule
reads a point's residuals, and a point no posed image observes has none; treating
"no finite residual" as the verdict without that guard would delete every point
of an unposed corner of the capture rather than leaving it alone.

**The frame rescale reads two different image tables.** The "before" distance is
measured against the input's poses and the "after" against the output's, because
the centroid itself moves when the cameras do. Measuring both against one table
would leave the frame carrying the gauge drift of the solve.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `releases` | empty: every camera held | One `CameraRelease` per camera-table entry: `focal` releases that camera's focal, `distortion` its `k1` or spline and needs `focal`. |
| `schedule` | `DEFAULT_SCHEDULE`, `[(50, 5), (12, 2), (4, 1)]` | The staged trim schedule, `(trim_px, loss_scale)` per round. |
| `max_iters` | `60` | LM iteration budget per round. |
| `min_track` | `2` | Trim survivors a point needs to stay in a round's solve. |
| `min_obs` | `12` | Trim survivors below which a round exits degenerate. |

The defaults after `releases` are the kernel's own
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)), stated
here so a caller sees what it is getting.

## Python bindings

`EditedReconstruction.bundle_adjust(*, opt_f=False, opt_distortion=False,
releases=None, schedule=None, max_iters=60, min_track=2, min_obs=12)` returns
`(EditedReconstruction, report)`. `opt_f` and `opt_distortion` are the release
given to every camera of the table. `releases`, when given, replaces them: a
list with one dict per camera in the table, in table order, each
`{"focal": bool, "distortion": bool}`, a missing key meaning `False`; an
unknown key or a value that is not a bool is a `ValueError`. It materialises the version's value when its
overlay is not empty, runs the function over it, and wraps the answer as a new
base with an empty overlay, so the Python surface is the viewer's edit exactly.
The report is the fields above as a dict, `cameras` a list with one dict per
camera in the solve carrying the `CameraAdjustment` fields (`outermost_observed` a dict of
`radius_px`, `theta_deg`, `image` and `xy`, or `None`), and every refusal is a
`ValueError` carrying the sentence the error writes.

```python
adjusted, report = value.bundle_adjust(opt_f=True)
# A two-camera rig: camera 0 held, camera 1's focal and spline released.
adjusted, report = value.bundle_adjust(
    releases=[{}, {"focal": True, "distortion": True}]
)
print(report["median_residual_before"], "->", report["median_residual_after"])
for camera in report["cameras"]:
    print(camera["camera"], camera["focal_before"], "->", camera["focal_after"])
```

## Testing

`crates/sfmtool-core/src/reconstruction/bundle_adjust/tests.rs` builds cameras on
a shallow arc looking at a cloud of points, with every observation at the exact
projection, so a perturbed copy of it has a known place to converge to; the
fixture needs no pixels, because the adjustment reads none. What it pins:

- The poses and the points converging, with three points held at the truth to
  pin the gauge the solve is otherwise free in, and the residual median coming
  down.
- The input value untouched.
- A released focal found from 6 % off, reported, and written into the camera.
- A value whose images are taken through two cameras adjusted rather than
  refused, each image read through its own lens: the poses converge, and each
  camera's report entry counts its images and holds its focal.
- A released spline moving toward a planted one with the focal, and a released
  `k1` on a `SIMPLE_RADIAL_FISHEYE` doing the same, each with the residual median
  falling by an order of magnitude and the report marking the camera's
  distortion released; a spline camera, a `k1` camera and a pinhole in one solve,
  the first two releasing their distortion and the third its focal alone; a
  spline camera released beside a pinhole held, which comes back unchanged.
- A rig of an `OPENCV_FISHEYE` camera and a spline camera, the first held and
  coming back unchanged while the spline moves, and a focal release on the
  `OPENCV_FISHEYE` refused naming it.
- Each per-camera refusal naming its camera in a one-line sentence: a distortion
  release without the focal, a distortion release on a `SIMPLE_PINHOLE`, a focal
  release on a `PINHOLE`.
- Every camera held, by an empty list and by an explicit one, giving the same
  solve bit for bit, the residual median falling and no lens moving.
- A release list one camera short and one camera long, refused.
- Each camera's outermost observation one of its own images', its radius
  measured from its principal point.
- The focal released on two cameras finding each its own planted focal and
  moving nothing else about either lens.
- A camera no posed image uses (one nothing references, one only an unposed
  image uses) coming back untouched, left out of the report, and its entry in
  the release list ignored even where its model could not take it.
- A point at infinity coming back a unit direction, and the infinity count with
  it.
- A held point coming back at exactly the coordinate it went in at, with its
  constraint.
- A point worn below `min_track` deleted, and `RowMap::by_scan` reporting the
  deletion and the survivors closing up.
- A patch frame scaling by the same ratio its placement distance did.
- Every refusal, by its own variant.

Bindings (`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real reconstruction -- the poses moving, the residual median
not getting worse, the report's populations and its per-camera entries, the value
that came back being a new base with no overlay, and the object it came from
untouched -- and the no-keypoints refusal.

## Non-goals

- Releasing the distortion of the multi-coefficient models (`RADIAL`,
  `OPENCV`, `OPENCV_FISHEYE`, …). The kernel has no rung exact for them; switch
  the camera to a spline model first
  ([`switch-camera-model.md`](switch-camera-model.md)).
- Deciding which points are at infinity. The representation the value carries is
  honoured for the whole solve; re-deciding it is
  [`../../cli/reconstruction/xform/find-points-at-infinity.md`](../../cli/reconstruction/xform/find-points-at-infinity.md).
- Adjusting a rig as a rig. Every image keeps its own free pose, whichever
  camera took it.
- Adding, removing or moving observations. The adjustment reads the track it is
  given.
- Running in the background. The function is synchronous, and a caller that
  needs a responsive window runs it where a stall is acceptable.
