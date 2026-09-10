# Moving one camera

An image's pose says where the photograph was taken from. When that answer is
wrong, everything computed from it is wrong with it, and the usual repair is to
re-estimate the pose from the correspondences — which lands the same answer
again whenever the correspondences are what is wrong. This spec describes the
other repair: putting the camera where a caller says it goes, and settling the
points that image observes around the new pose.

The function is pure. It takes the reconstruction value and a pose, and returns
a new value plus a report; the input is left exactly as it was whether the move
succeeds or is refused.

The pose is the caller's decision and the points are its consequence, and that
ordering is what makes this different from an estimator. Nothing here judges
whether the pose is a good one, and no single track's failure to re-solve
refuses the move: a track that will not triangulate at the new pose keeps the
position it had and is counted, because the alternative would be a call that
refuses a placement the caller already made over a point that was never the
question.

Related specs:
[`remove-observation.md`](remove-observation.md) (the other edit that re-solves
a track from the rays that remain, and the ray and triangulation helpers this
shares with it), [`bundle-adjust.md`](bundle-adjust.md) (the patch-frame rescale
and the stored-error rewrite this shares with it, and the adjustment that moves
*every* pose), [`edited-reconstruction.md`](edited-reconstruction.md) (the value,
and the row map a caller follows an index through), and
[`../../gui/edits/move-camera.md`](../../gui/edits/move-camera.md) (the viewer's
edit over it, where the pose comes from a hand on the viewport).

## Rust API

The function lives in
[move_camera.rs](../../../crates/sfmtool-core/src/reconstruction/move_camera.rs),
re-exported as `sfmtool_core::{move_camera, MoveCameraReport, MoveCameraError}`
and bound as `EditedReconstruction.move_camera`.

```rust
pub fn move_camera(
    recon: &SfmrReconstruction,
    image: usize,
    world_from_camera: &Se3Transform,
) -> Result<(SfmrReconstruction, MoveCameraReport), MoveCameraError>;

pub struct MoveCameraReport {
    pub image: usize,
    pub rotation_deg: f64,
    pub translation: f64,
    pub translation_scene: Option<f64>,
    pub observed: usize,
    pub retriangulated: usize,
    pub kept: usize,
    pub rotated_bearings: usize,
    pub residual_before_px: Option<[f64; 2]>,
    pub residual_after_px: Option<[f64; 2]>,
}

pub enum MoveCameraError {
    ImageOutOfRange { image: usize, image_count: usize },
    NoPose(usize),
    InvalidPose,
}
```

**The pose is world-from-camera**: the rotation carries camera axes onto world
axes, and the translation *is* the camera centre. That is the direction a caller
holding a camera thinks in — "put it here, pointing that way" — while the value
stores the inverse, and converting once here is better than making every caller
do it. `Se3Transform` rather than a rigid
transform because it is the type a viewer's own transform arithmetic produces
when it divides a display transform out of a viewport pose; a camera pose has no
scale, so `Se3Transform::scale` is not read.

**One report, not several returns**, and every number in it is one the caller
has no other way to compute: how far the camera moved, in the reconstruction's
units and in the capture's own, and what became of each track that observes it.
The three track counts sum to `observed`, so a caller can check its own
arithmetic against the call's.

**The residual pair is the median and 90th percentile** of this image's own
reprojection residuals, in pixels, before and after — over the image's
observations rather than over the whole value, because it is this image that
moved. `None` where the value carries no inline keypoints, since there is then
no pixel to reproject against. Half a photograph agreeing is not the same claim
as all of it agreeing, and it is the tail that says which, so the pair travels
together.

**The errors are an enum** because the caller is a menu entry that has to say in
one sentence what did not hold, and "the image is not there", "the image has no
pose to replace" and "that is not a pose" are three different mistakes with
three different fixes.

```rust
use sfmtool_core::reconstruction::move_camera::{move_camera, pose_of};
use sfmtool_core::{RotQuaternion, Se3Transform};

// Where image 7 stands now, and half a metre to its left.
let stored = pose_of(&recon, 7);
let moved = Se3Transform::new(
    stored.rotation.clone(),
    stored.translation + nalgebra::Vector3::new(-0.5, 0.0, 0.0),
    1.0,
);
let (next, report) = move_camera(&recon, 7, &moved)?;
assert_eq!(report.retriangulated + report.kept + report.rotated_bearings, report.observed);
```

Three helpers travel with it, because a caller steering a camera by hand needs
the same residual the report carries, for a pose the value does not hold yet:

```rust
/// Where an image stands, in the form `move_camera` takes.
pub fn pose_of(recon: &SfmrReconstruction, image: usize) -> Se3Transform;

/// One observation as the residual measures it.
pub struct ReprojectionSample {
    pub position: Point3<f64>,
    pub at_infinity: bool,
    pub keypoint: [f64; 2],
}

pub fn image_reprojection_samples(
    recon: &SfmrReconstruction,
    image: usize,
) -> Vec<ReprojectionSample>;

/// The same over an edited value, which is what a caller holding a version has.
pub fn edited_image_reprojection_samples(
    edited: &EditedReconstruction,
    image: usize,
) -> Vec<ReprojectionSample>;

/// The median and 90th percentile, in pixels, of `samples` under a pose.
pub fn residual_quantiles_px(
    camera: &CameraIntrinsics,
    world_from_camera: &Se3Transform,
    samples: &[ReprojectionSample],
) -> Option<[f64; 2]>;
```

The split is what lets the samples be gathered **once** and measured against a
pose that changes many times a second: nothing in the value moves while a caller
is choosing a pose, so the points and the pixels are fixed and only the
projection is repeated.

## What the call does

1. **The pose is replaced.** The row's world-to-camera rotation becomes the
   inverse of the given rotation, and its translation the vector that carries
   the given centre onto the camera origin.

2. **Every track this image observes is settled**, each on its own evidence:

   - a **finite point with two or more observations that all carry a pixel** is
     re-triangulated from the value's own poses and lenses, the moved one
     included, by the same solve a shortened track goes through in
     [`remove-observation.md`](remove-observation.md). Its patch frame is
     rescaled by the ratio of its placement distances, so the patch keeps the
     angular size it had, and its stored `error` becomes the RMS of its own
     residuals at the new geometry, because the column would otherwise describe
     a geometry the value no longer holds;
   - a **bearing whose only observation is this image** rotates with the camera:
     its stored direction becomes the moved camera's ray through its keypoint.
     One ray is exactly what a bearing is, and it is the ray that moved;
   - a **bearing more than one image sees** keeps its direction. Its rays now
     disagree about a rotation nothing asked them for, and picking one of them
     would be an estimate rather than a consequence;
   - a **track whose triangulation fails** — near-parallel rays, or a solve
     behind one of the cameras — keeps its position, and is counted;
   - a **track whose observations carry no pixel** keeps its position. A
     `sift_files` value without the format's optional inline keypoint column
     states no ray at all, so its cameras move and its points stand.

3. **The derived indexes are rebuilt**, so the value that comes back is
   complete.

**Refusals.** The image index is past the table; the image carries no pose, its
stored rotation or translation being a non-finite placeholder rather than a
registration; or the pose is not a finite rigid placement.

## Theory

### Why the observed points move and nothing else does

A pose enters the reconstruction in exactly two ways: as the frame a point was
triangulated in, and as the camera a residual is measured against. Moving one
pose therefore invalidates precisely the points that pose helped place — the
ones this image observes — and nothing else. A point no other image sees is a
point whose position was a claim about where this camera was, and the claim has
changed; a point this image never observed was placed without it, and is
untouched by construction.

That is also what makes the un-observed points the honest evidence about whether
a pose is right. They were placed by other cameras, they do not care where this
one is, and the photograph either lands on them or does not.

### The two representations, and why only one of them turns

A finite point states a location; a bearing states a direction. A location is
recovered from two or more rays, so re-triangulating it at the new pose is the
same operation that produced it in the first place. A direction is one ray, and
a single-view bearing's one ray *is* this camera's — so the direction turns with
the camera exactly and without a solve. Two rays that both state a direction are
a different matter: they were reconciled at the old geometry, and reconciling
them again is an estimate this call does not make.

### The residual pair, before and after

The "before" pair is measured at the stored pose over the stored positions; the
"after" pair at the new pose over the positions that came back. They are not two
measurements of the same thing, and they are not meant to be: the first says how
well the photograph agreed with the structure as it stood, the second how well
it agrees with the structure the move produced. A move that lines the photograph
up brings the first number down. A move that merely drags every observed point
along with the camera leaves it flat, which is the signal a caller wants — and
the reason the pair is reported for the moved image alone rather than for the
whole value, where the other cameras' agreement would drown it.

### The capture's own length unit

`translation` is in the reconstruction's coordinates, which mean nothing on
their own: a solve is up to a similarity. `translation_scene` divides it by the
median over images of that image's median camera-to-structure distance, which is
the same unit the resection reports its displacements in
([`../../gui/resect-image.md`](../../gui/resect-image.md)), so "a tenth of
the scene" means the same thing in both. It is `None` for a reconstruction with
no finite structure to measure against, where every displacement is unitless.

## Implementation notes

**Four helpers are shared rather than copied.** The pixel-to-world ray is
`ImageTable::world_ray`, the single-track solve with its three refusal signals is
`triangulation::triangulate_track`, the patch-frame resize is
`bundle_adjust::rescale_patch_frame` and the per-observation residual is
`data::observation_reprojection_error`. Each has exactly one definition in the
crate, so this call and the edits beside it cannot come to disagree about what a
ray is, which solves are acceptable, or how a patch keeps its angular size.

**The rays are built at the new poses, including for the observations of other
images.** Only one image moved, so every other ray is the ray it was — but
reading them all out of the output value rather than the input one means there
is no second statement of which poses the solve is against.

**The placement scale is read on both sides of the move.** A patch frame's
angular extent converts to a world one through the distance from the camera-cloud
centroid, and moving a camera moves that centroid. The rescale therefore measures
`before` against the input's image table and `after` against the output's, which
is the same treatment the adjustment gives a solve that moved every camera.

**The report's counts partition the observed tracks.** `retriangulated + kept +
rotated_bearings == observed` holds by construction: each track takes exactly one
of the three branches. A caller can use it as a check on its own reading of what
happened.

## Python bindings

`EditedReconstruction.move_camera(image, quaternion_wxyz, translation) ->
(EditedReconstruction, report)`, in
[edited.rs](../../../crates/sfmtool-py/src/reconstruction/edited.rs). The pose is
the same world-from-camera pair the Rust call takes, as a WXYZ list and a
three-vector; the quaternion is normalised on arrival. A **bulk** edit, so the
value that comes back is a whole new base with an empty overlay, and the object
called on is unchanged.

The report is a dict carrying `image`, `rotation_deg`, `translation`,
`translation_scene`, `observed`, `retriangulated`, `kept`, `rotated_bearings`,
`residual_before_px` and `residual_after_px`; the last two are two-element float
arrays, or `None` where the value carries no inline keypoints. A refusal is a
`ValueError` carrying the error's own sentence.

```python
recon = SfmrReconstruction.load(path)
value = EditedReconstruction(recon.to_embedded_patches())
after, report = value.move_camera(7, [1.0, 0.0, 0.0, 0.0], [1.5, -0.2, 0.9])
print(report["rotation_deg"], report["retriangulated"])
```

## Testing

`crates/sfmtool-core/src/reconstruction/move_camera/tests.rs`, over a synthetic
scene of four cameras whose stored keypoints are the exact projections of known
points, with one camera displaced so that moving it back has a known right
answer: every observed point returns to where the truth solve put it, the input
is untouched, a single-view bearing becomes the moved camera's ray, a multi-view
bearing and a keypoint-less track keep their positions and are counted, a
triangulation that fails is counted and its point unmoved, a re-triangulated
point's frame is rescaled by its depth ratio, the stored error is rewritten, the
report's counts partition the observed tracks, the residual pair falls to zero
at the truth pose, and each refusal.

`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py` covers the
call's shape over a real reconstruction: a camera put back where it stands
changes nothing but the value, a moved one moves its own pose and no other, and
the two refusals.

## Non-goals

- **Estimating the pose.** That is
  [`../../gui/resect-image.md`](../../gui/resect-image.md), which computes
  a pose from correspondences; this one is told the pose.
- **Refining anything else.** No other pose, no lens, and no point this image
  does not observe moves. Running the adjustment afterwards is
  [`bundle-adjust.md`](bundle-adjust.md), which is where the other cameras get
  to answer.
- **Moving several cameras at once**, or a rig as a unit. One call, one image.
- **Re-solving a multi-view bearing.** Its direction is kept; converting a
  bearing back to a finite point is what a second observation does, in
  [`add-observation.md`](add-observation.md).
