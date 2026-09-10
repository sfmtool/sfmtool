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
takes poses, points and observations as plain arrays over one shared camera.
What this adds is the layer between that kernel and a reconstruction: which
images are posed, which observations carry a pixel, what each point's stored
constraint means, and how the answer goes back into a value whose patch frames,
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
[`add-observation.md`](add-observation.md) and
[`remove-observation.md`](remove-observation.md) (the two edits whose patch-frame
rescale this one uses), and
[`../../gui/edits/bundle-adjust.md`](../../gui/edits/bundle-adjust.md) (the
viewer's edit over it).

## Rust API

The function lives in
[bundle_adjust.rs](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust.rs),
re-exported as `sfmtool_core::{bundle_adjust, BundleAdjustOptions,
BundleAdjustReport, BundleAdjustError}` and bound as
`EditedReconstruction.bundle_adjust`.

```rust
pub fn bundle_adjust(
    recon: &SfmrReconstruction,
    options: &BundleAdjustOptions,
) -> Result<(SfmrReconstruction, BundleAdjustReport), BundleAdjustError>;

pub struct BundleAdjustOptions {
    pub opt_f: bool,
    pub schedule: Vec<BaSchedule>,
    pub max_iters: usize,
    pub min_track: usize,
    pub min_obs: usize,
}

pub struct BundleAdjustReport {
    pub images: usize,
    pub points: usize,
    pub observations: usize,
    pub points_deleted: usize,
    pub median_residual_before: f64,
    pub median_residual_after: f64,
    pub focal_before: f64,
    pub focal_after: f64,
    pub focal_released: bool,
}

pub enum BundleAdjustError {
    NoKeypoints,
    MixedCameras { cameras: usize },
    FocalNotReleasable(&'static str),
    NoPosedImages,
    NoObservations,
    EmptySchedule,
    Constraints(PointConstraintsError),
    Degenerate { observations: usize, min_obs: usize },
}

/// Whether the kernel's analytic focal column is exact for this camera.
pub fn focal_is_releasable(camera: &CameraIntrinsics) -> bool;
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
whether the lens moves; everything else is the schedule the rest of the toolkit
runs. So `opt_f` is the first field and the rest are the kernel's own defaults,
stated rather than hidden so a caller that does need to tighten a trim can.

**`opt_f` refuses rather than degrades.** The kernel silently reduces a focal
release on a model its analytic focal column is not exact for to a fixed-focal
solve. That is right for a kernel with a parity requirement and wrong for a
caller who asked a question: a report saying the focal was released when it was
held would be false. [`focal_is_releasable`] is public for the same reason, so a
caller offering the release as a choice can grey the choice instead of taking it
and refusing.

**The report names three populations.** `images`, `points` and `observations`
are what went into the solve, which is not the whole reconstruction: an unposed
image is not in it, and neither is a point nothing posed observes.
`points_deleted` is what came out short. A caller reporting the edit in one line
needs all four to write a true sentence.

**The errors are an enum.** Every variant is a property of the value or the
options that a caller could have checked, and two of them (`NoKeypoints`,
`MixedCameras`) are exactly what a menu entry greys itself on. A
`Result<_, String>` would make that gate a string comparison.

### Example

```rust
use sfmtool_core::{bundle_adjust, BundleAdjustOptions};

let options = BundleAdjustOptions {
    opt_f: true,
    ..BundleAdjustOptions::default()
};
let (next, report) = bundle_adjust(&recon, &options)?;
println!(
    "{} px -> {} px, focal {} -> {}, {} points dropped",
    report.median_residual_before,
    report.median_residual_after,
    report.focal_before,
    report.focal_after,
    report.points_deleted,
);
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
- Posed images that do not share one set of camera intrinsics
  (`MixedCameras`). The kernel carries a single shared camera, and a value whose
  images disagree about the lens has to be told so rather than adjusted through
  one of them.
- A focal release on a model the kernel's focal column is not exact for
  (`FocalNotReleasable`).
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
spelling of the reprojection in this module.

Both medians are taken over the finite residuals of the points that survive, so
the two numbers describe one population and their difference is the improvement
on it.

### 4. Writing the answer back

- **Poses.** Each posed image takes the rotation and translation the solve
  ended with.
- **The focal.** Under `opt_f`, the shared camera is replaced by itself at the
  solved focal. Nothing else about the lens moves.
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
`SfmrReconstruction::materialize_points_at_infinity` places a bearing and what
[`add-observation.md`](add-observation.md) multiplies by and
[`remove-observation.md`](remove-observation.md) divides by when a point crosses.
It is `ImageTable::placement_scale`, one function for all three, so their
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
| `opt_f` | `false` | Release the shared focal length. |
| `schedule` | `DEFAULT_SCHEDULE`, `[(50, 5), (12, 2), (4, 1)]` | The staged trim schedule, `(trim_px, loss_scale)` per round. |
| `max_iters` | `60` | LM iteration budget per round. |
| `min_track` | `2` | Trim survivors a point needs to stay in a round's solve. |
| `min_obs` | `12` | Trim survivors below which a round exits degenerate. |

The defaults after `opt_f` are the kernel's own
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)), stated
here so a caller sees what it is getting.

## Python bindings

`EditedReconstruction.bundle_adjust(*, opt_f=False, schedule=None,
max_iters=60, min_track=2, min_obs=12)` returns
`(EditedReconstruction, report)`. It materialises the version's value when its
overlay is not empty, runs the function over it, and wraps the answer as a new
base with an empty overlay, so the Python surface is the viewer's edit exactly.
The report is the fields above as a dict, and every refusal is a `ValueError`
carrying the sentence the error writes.

```python
adjusted, report = value.bundle_adjust(opt_f=True)
print(report["median_residual_before"], "->", report["median_residual_after"])
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
not getting worse, the report's populations, the value that came back being a new
base with no overlay, and the object it came from untouched -- and the
no-keypoints refusal.

## Non-goals

- Releasing the distortion parameters. The kernel's `opt_k1` and `opt_bspline`
  rungs are not exposed here; a caller staging those runs the kernel directly.
- Deciding which points are at infinity. The representation the value carries is
  honoured for the whole solve; re-deciding it is
  [`../../cli/reconstruction/xform/find-points-at-infinity.md`](../../cli/reconstruction/xform/find-points-at-infinity.md).
- Adjusting a rig as a rig, or several cameras at once. One shared camera, or a
  refusal.
- Adding, removing or moving observations. The adjustment reads the track it is
  given.
- Running in the background. The function is synchronous, and a caller that
  needs a responsive window runs it where a stall is acceptable.
