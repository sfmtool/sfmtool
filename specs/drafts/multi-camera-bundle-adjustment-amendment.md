# Bundle adjustment over several cameras

**Status:** Draft. Amends
[`core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
(the array kernel) and
[`core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
(the reconstruction-level function), and through them
[`gui/edits/bundle-adjust.md`](../gui/edits/bundle-adjust.md). Decided: the
interface below, the per-camera lens blocks, the release rule, the report, and
the model-aware in-front test for directions. Open: nothing that blocks the
build; see "Open questions".

## What changes, in one paragraph

sfmtool's bundle adjustment refines every pose and point of a reconstruction
against its observations, but only when every posed image shares one set of
camera intrinsics: the kernel takes a single `CameraIntrinsics`, and the
reconstruction-level function refuses a value whose images disagree about the
lens (`MixedCameras`). A capture from a rig, or from two phones, or a reconstruction
whose cameras were calibrated separately, has several. This amendment lets the
kernel take a list of cameras and, per image, the index of the camera that took
it. Each camera keeps its own lens parameters in the solve; the released ones
(focal, `k1`, spline) are released per camera, not shared across cameras; and
the reconstruction-level function adjusts a value with any number of cameras
instead of refusing it.

A camera here is what the `.sfmr` calls one: an entry of the camera table,
which images reference by index. Rigs are not modelled: every image keeps its own
free pose, as it does today.

## The kernel

### Interface

```rust
/// The cameras of a solve, and which of them took each image.
pub struct BaCameras<'a> {
    /// One entry per camera; each carries its own model, parameters and
    /// initial focal.
    pub cameras: &'a [CameraIntrinsics],
    /// One entry per image (`n_img`): the index into `cameras` of the camera
    /// that took it.
    pub image_camera: &'a [u32],
}

impl<'a> BaCameras<'a> {
    /// Every image taken by one camera: the single-camera case.
    pub fn shared(camera: &'a CameraIntrinsics, n_img: usize) -> BaCameras<'a>;
}

pub fn bundle_adjust(
    cameras: &BaCameras<'_>,             // replaces `cam: &CameraIntrinsics`
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    point_at_infinity: Option<&[bool]>,
    constraints: Option<&PointConstraints>,
    free_points: FreePointPolicy,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
    progress: &Progress<'_>,
) -> BundleAdjustment;

pub struct BundleAdjustment {
    /// The cameras after the solve, one per input camera, in order: each the
    /// input camera with its released parameters replaced by the solved ones
    /// (focal under `opt_f`, `k1` under `opt_k1`, the spline under
    /// `opt_bspline`), and unchanged otherwise.
    pub cameras: Vec<CameraIntrinsics>,
    pub residual_norms: Vec<f64>,
    pub point_at_infinity: Vec<bool>,
}
```

**Why `BaCameras` rather than two more arguments.** The camera list and the
image-to-camera column only mean something together, and the checks on them
(every index in range, `image_camera.len() == n_img`) belong in one place.
`BaCameras::shared` keeps every single-camera caller a one-line change:
`&BaCameras::shared(&cam, quats.len())` where `&cam` was.

**Why the result returns cameras.** The scalar `focal`, `k1` and `bspline` fields
describe one camera. With several, each has its own, and a caller wants the
camera back to write into its own table, which is what every current caller
already builds from those three fields. The result carries the whole
`CameraIntrinsics` per camera instead, so a caller copies it rather than
reassembling it. This removes `focal`, `k1` and `bspline` from the struct; the
current callers (`reconstruction_growth`, `rotation_init`, the
reconstruction-level function and the Python binding) read `cameras[0]`.

**Validation.** `image_camera.len() != quats.len()`, an index past
`cameras.len()`, or an empty `cameras` is a caller error and panics like the
kernel's other shape checks; the Python binding checks the same and raises
`ValueError`.

### The reduced camera system

Today the reduced system is `[f?, k1?, c₀..c_{N−1}? | 6·n_img]`: one block of
lens slots, shared. With several cameras it is one lens block per camera, then
the poses:

```
[ f₀?, k1₀?, c₀,₀..c₀,N₀−1? | f₁?, k1₁?, c₁,₀..c₁,N₁−1? | … | 6·n_img ]
```

Camera `j`'s block has a focal slot, a `k1` slot and `N_j` spline slots, where
`N_j` is camera `j`'s own spline length (`0` for a model without one). An
observation of image `i` writes its lens columns into the block of
`image_camera[i]` and nothing into any other camera's block, so two cameras'
lens parameters couple only through the poses and points they both observe.

Every rule the single block follows today applies to each block separately:

- **Release gates, per camera.** `opt_f`, `opt_k1` and `opt_bspline` are requests
  to every camera. A camera whose model the release is not exact for keeps that
  parameter fixed (the core's existing degrade, now decided per camera), so a
  pinhole and an `OPENCV_FISHEYE` in one solve under `opt_f` release the
  pinhole's focal and hold the fisheye's.
- **Pinning.** A slot that is not released, or is released with no surviving
  observation touching it, is pinned with an identity row and column and a zero
  gradient entry. A camera none of whose images has a surviving observation in a
  round therefore has its whole block pinned for that round.
- **Step guards.** A candidate step is rejected, and re-damped, when any camera's
  candidate focal is non-positive, its `k1` folds inside its own imaged field, or
  its spline breaks monotonicity on its own domain. Each camera is checked against
  its own model and field.
- **Staged releases** are the caller's schedule, as today.

Width: `Σ_j (2 + N_j) + 6·n_img`. With a handful of cameras and a few hundred
images the dense solve is unchanged in practice.

### What reads the camera, per observation

Everything that read the one camera now reads the camera of the observation's
image, `cameras[image_camera[obs_img[k]]]`:

- the projection, its Jacobian and the lens columns;
- the trim's finite in-front floor `1e-3 · f`, with `f` that camera's focal (the
  mean of its two where the model carries two);
- `pixel_to_ray` in the inter-round re-estimation, so a track seen by two cameras
  back-projects each observation through its own lens.

**The noise floor** under `FreePointPolicy::cross`, `θ_floor = c · s / f`, is an
angle per track. A track seen through several cameras takes `f` as the mean of the
focal lengths of its observations' cameras, one term per observation, which is the
single-camera rule when every observation is through one camera.

### In front, for directions

The finite in-front measure is already model-aware: the canonical depth `−z_cam`
for the perspective family, and the range `‖p_cam‖` for a ray-path model, whose
domain runs past `θ = 90°` (see "Model-aware in-front measure" in the kernel). A
direction today uses `−(R·d)_z` for every model, so a point at infinity observed
more than 90° off-axis through a fisheye wider than 180° is trimmed as if it were
behind the camera. The rule becomes the same as for finite points: for a ray-path
model a direction is in front when `ray_to_pixel(R·d)` is defined, and for the
perspective family when `(R·d)_z < 0`. The same test applies wherever the kernel
asks whether a direction is in front: the trim, and the penalized-residual branch
of the solve.

### Parity

With one camera and every image on it, the kernel is the one it is today, bit for
bit, on every output (poses, points, residual norms, representation, and the
camera's focal, `k1` and spline). The direction in-front change is the one
exception: on a ray-path model it keeps observations of directions past 90°
off-axis that are trimmed today, and a parity test covers the perspective family
and ray-path scenes with no direction past 90°.

## The reconstruction-level function

`bundle_adjust(recon, options, progress)` keeps its signature. What changes:

- **No `MixedCameras` refusal.** The solve gathers the camera table's entries
  that posed images use, re-indexed onto that subset in table order, and the
  per-image camera index, and passes them as a `BaCameras`. An unposed image's
  camera, if no posed image uses it, is not in the solve and comes back exactly
  as it went in.
- **The release refuses rather than degrades, over every camera.** Under
  `opt_f`, the call refuses with `FocalNotReleasable` when any camera in the
  solve has a model the focal column is not exact for. The error names the
  camera index and its model. A caller that wants to release some cameras and not
  others is out of scope (see "Open questions").
- **Writing back.** Each camera in the solve is replaced by the kernel's returned
  camera, so under `opt_f` each gets its own solved focal and nothing else about
  any lens moves.
- **The report** replaces the single `focal_before`, `focal_after` and
  `focal_released` with one entry per camera in the solve:

  ```rust
  pub struct BundleAdjustReport {
      pub images: usize,
      pub points: usize,
      pub observations: usize,
      pub points_deleted: usize,
      pub median_residual_before: f64,
      pub median_residual_after: f64,
      pub cameras: Vec<CameraAdjustment>,
  }

  pub struct CameraAdjustment {
      /// The camera's index in the reconstruction's camera table.
      pub camera: usize,
      /// Posed images taken by it that were in the solve.
      pub images: usize,
      pub focal_before: f64,
      pub focal_after: f64,
      pub focal_released: bool,
  }
  ```

  Per-camera residual medians are not in the report; the overall medians are the
  ones a caller reports in one line, and a caller wanting more runs the kernel.
- **`focal_is_releasable`** stays public and per camera. A caller offering the
  release greys it unless every camera the posed images use passes.

## Callers

- **The Python bindings.**
  - `geometry.bundle_adjust` takes `cameras=[...]` and `image_camera=` (an
    `(n_img,)` integer array) in place of its single `camera`. It returns
    `cameras` (the solved intrinsics, one per input camera) in place of `focal`,
    `k1` and `bspline_coefficients`. The binding keeps no single-camera alias
    for the old arguments.
  - `EditedReconstruction.bundle_adjust` keeps its arguments. Its report dict
    takes a `cameras` list of per-camera dicts in place of the three focal keys.
- **SfM Explorer.** The Bundle Adjust edit
  ([`gui/edits/bundle-adjust.md`](../gui/edits/bundle-adjust.md)) stops greying on
  mixed cameras. "Release focal" is greyed unless every camera the posed images
  use is releasable, with the hover reason naming the first that is not. The
  Action Log line reports a focal change per camera. The MCP tool's
  `release_focal` and its report follow.
- **`reconstruction_growth` and `rotation_init`** pass `BaCameras::shared` and read
  `cameras[0]`; their behaviour does not change.
- **`sfm xform --bundle-adjust`** keeps its COLMAP round trip; switching it to the
  native adjustment is a separate decision.

## Testing

Kernel:

- **Parity.** Every existing single-camera test passes unchanged through
  `BaCameras::shared`, and a single-camera scene run through an explicit
  one-entry `BaCameras` matches it to the bit.
- **Two cameras, different models.** A scene with a `PINHOLE` and an
  `OPENCV_FISHEYE` camera, images from each, and perturbed poses and points,
  converges to sub-pixel reprojection. The same scene with the fisheye's images
  projected through the pinhole (the wrong lens) does not, which shows each
  observation is read through its own camera.
- **Per-camera focal release.** Two releasable cameras with different planted
  focal lengths, both started several percent off, recover each its own focal
  under `opt_f`. With one releasable and one not, the releasable one moves and
  the other comes back bit for bit.
- **Independent blocks.** An observation's lens columns land only in its own
  camera's block: a camera with no observations comes back bit for bit under every
  release.
- **Directions past 90°.** On an equidistant fisheye scene wider than 180°, a
  direction observed at `θ = 100°` survives the trim and constrains the rotation;
  on a perspective scene a direction behind the camera is still trimmed.

Reconstruction-level:

- A value with two cameras is adjusted rather than refused; poses and points
  converge; each camera's report entry is correct.
- `opt_f` with every camera releasable releases each; with one not releasable it
  refuses naming that camera.
- A camera no posed image uses comes back untouched.

Bindings: the new `cameras` / `image_camera` arguments and results, their
`ValueError`s, and the per-camera report of `EditedReconstruction.bundle_adjust`.

Validation on real data: the Kerry Park reconstruction
(`kerry_park_ground_truth_candidate_tk105.sfmr`: 48 images from two
`OPENCV_FISHEYE` cameras of a rig, 400 points, 7 of them at infinity) is adjusted
without refusal and without releasing the lens. Checks:

- the median residual falls;
- no point is deleted that the trim cannot account for, including the 16 points
  COLMAP's adjustment drops because every one of their observations is more than
  90° off its camera's axis;
- the poses stay close to the input's;
- the result is compared with COLMAP's adjustment of the same value, after a
  similarity alignment of the camera centres.

## Open questions

- **Releasing some cameras and not others.** The release flags apply to every
  camera. A per-camera release (for example, one calibrated camera held while
  another is released) needs a per-camera flag in both the kernel and the
  options; nothing here precludes adding it.
- **Rigs.** A rig's cameras move together, and a rig-aware adjustment would solve
  one pose per frame plus fixed or refined camera-to-rig offsets. That is a
  different parameterization, not a consequence of several cameras, and is not
  proposed here.
