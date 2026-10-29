# Staged bundle adjustment

## Purpose

Staged bundle adjustment jointly refines camera poses and 3D points, and
optionally each camera's focal length and lens distortion (a radial coefficient
or a spline), so that the points project as closely as possible onto the pixels
where they were observed. It works in rounds. Each round first drops the
observations whose reprojection error is above that round's threshold, then
minimizes a robust (soft-L1) pixel reprojection error over what remains; every
round after the first also re-triangulates the points from the current poses
before it trims. The default schedule tightens the threshold from 50 px to
12 px to 4 px, so gross outliers are removed before the fine fit. The images
may be taken through one or more cameras: each camera keeps its own lens
parameters in the solve, and every observation is read through the camera of
its own image, with that image's intrinsic adjustment applied where it has one:
held (see "Held image adjustments"), or solved under a prior centred on the mean
of the camera's images where the caller releases it (see "The per-image
release").

This is the optimizer that the trimmed pose-only refinement
(`crates/sfmtool-core/src/geometry/pose_refine.rs`) is the single-pose
special case of. It runs inside [reconstruction growth](reconstruction-growth.md)
and [far-field rotation initialization](rotation-init.md), and the
[reconstruction-level adjustment](../reconstruction/bundle-adjust.md) calls it
on a whole `.sfmr` for the viewer's Bundle Adjust command and for
`sfm xform --bundle-adjust` when any camera of the reconstruction has a spline
model (otherwise that command uses pycolmap).

## Definitions

- `n_cam` **cameras**, each a `CameraIntrinsics` with its own model,
  parameters and initial focal. A camera here is what the `.sfmr` calls one:
  one of the reconstruction's cameras, which images reference by index. Rigs are not
  modelled: every image keeps its own free pose.
- `n_img` **images**, each taken through one of the cameras (`image_camera[i]`),
  optionally with an intrinsic adjustment of that camera's focal and principal
  point (`image_adjustments[i]`), and each with a
  world-to-camera pose `(R_i, t_i)` in the canonical convention
  (`x_cam = R·X + t`; the camera looks along `−Z`, a point in front has
  `z < 0`), rotations supplied as WXYZ unit quaternions.
- `n_pt` world **points** `X_p` (canonical world frame). Points may be
  non-finite (`NaN`) — their observations are invalid until a
  retriangulation round replaces them.
- `n_obs` **observations** `(image, point, uv)` with `uv` the observed full
  (un-centered) pixel position.
- A **track** is the set of observations of one point.

The state arrays are full-sized (the solve compacts internally over what
the observations reference). Images never touched by an observation pass
through unchanged. Points do too under a single-round schedule — but any
retriangulation round (rounds after the first) rebuilds the whole points
array from the supplied observations, so under a multi-round schedule an
unobserved point comes back `NaN`, not unchanged (see step 1 below; the
callers refill).

## The staged loop

The kernel lives in
[bundle_adjust.rs](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs)
(`bundle_adjust`, `BaCameras`, `BaSchedule`, `BundleAdjustment`,
`CameraRelease`, `ImagePriorOptions`, `ImagePriorWidths`, `ImagePrior`,
`PointConstraint`, `PointConstraints`, `DistanceReference`, `FreePointPolicy`,
`FreePointDecision`), bound as `sfmtool.geometry.bundle_adjust`.

```rust
pub struct BaSchedule {
    pub trim_px: f64,     // pre-round trim threshold on the residual norm; may be ∞
    pub loss_scale: f64,  // soft-L1 scale for the round's solve, px
}

/// The cameras of a solve, and which of them took each image.
pub struct BaCameras<'a> {
    pub cameras: &'a [CameraIntrinsics], // n_cam, each with its initial focal
    pub image_camera: Cow<'a, [u32]>,    // n_img, an index into `cameras`
    pub releases: Option<&'a [CameraRelease]>, // n_cam; None = the flags alone
    pub image_adjustments: Option<Cow<'a, [IntrinsicAdjustment]>>, // n_img; None = none
}

/// What one camera may release; `Default` and `HELD` release nothing.
pub struct CameraRelease {
    pub focal: bool,
    pub distortion: bool,      // k1 or the spline, whichever the model carries
    pub principal_point: bool, // (cx, cy), on any model
    pub image_focal: bool,           // each image's focal scale; holds the shared focal
    pub image_principal_point: bool, // each image's offset; holds the shared principal point
}

/// The widths of one camera's prior on its images' adjustments.
pub struct ImagePriorWidths {
    pub focal: f64,           // the log focal scale ln(1 + s)'s, read as a fraction
    pub principal_point: f64, // the offset's, in the short-side unit
}

pub struct ImagePriorOptions {
    pub loose: ImagePriorWidths,         // the first round's, and the cap (0.1, 0.1)
    pub floor: ImagePriorWidths,         // the least a derived width may be (0.001, 0.001)
    pub fixed: Option<ImagePriorWidths>, // every round's, derived from nothing
}

/// The prior one camera's per-image release ran under.
pub struct ImagePrior {
    pub widths: ImagePriorWidths, // the final round's; NaN where not released per image
    pub derived: bool,            // false under `fixed`
    pub spread: ImagePriorWidths, // the last derivation's, before the clamp
}

impl<'a> BaCameras<'a> {
    /// Every image taken by one camera: the single-camera case.
    pub fn shared(camera: &'a CameraIntrinsics, n_img: usize) -> BaCameras<'a>;
}

pub fn bundle_adjust(
    cameras: &BaCameras<'_>,             // the cameras and the image-to-camera column
    quats: &mut [UnitQuaternion<f64>],   // n_img, world-to-camera
    trans: &mut [Vector3<f64>],          // n_img
    points: &mut [[f64; 3]],             // n_pt (NaN allowed)
    uv: &[[f64; 2]],                     // n_obs
    obs_img: &[u32],                     // n_obs
    obs_pt: &[u32],                      // n_obs
    point_at_infinity: Option<&[bool]>,  // n_pt, the INITIAL representation
    constraints: Option<&PointConstraints>,  // n_pt constraints; None = all free
    free_points: FreePointPolicy,        // default: free points cross; NO_CROSS opts out
    protected: Option<&[bool]>,          // n_obs
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    opt_pp: bool,
    opt_image_f: bool,
    opt_image_pp: bool,
    image_prior: Option<&ImagePriorOptions>, // None = ImagePriorOptions::default()
    schedule: &[BaSchedule],             // default 50/5 → 12/2 → 4/1
    max_iters: usize,                    // LM iterations per round
    min_track: usize,                    // trim survivors per point (2)
    min_obs: usize,                      // degenerate-exit floor (12)
    progress: &Progress<'_>,             // phases, counts and the cancel flag
) -> BundleAdjustment;

pub struct BundleAdjustment {
    pub cameras: Vec<CameraIntrinsics>,  // n_cam, the cameras after the solve
    pub residual_norms: Vec<f64>,        // n_obs
    pub point_at_infinity: Vec<bool>,    // n_pt, the representation each ended with
    pub free_point_decision: Option<FreePointDecision>, // the storage decision; None under NO_CROSS or on a degenerate exit
    pub image_adjustments: Vec<IntrinsicAdjustment>, // n_img, after the solve
    pub image_priors: Vec<Option<ImagePrior>>,       // n_cam; None without a per-image release
}
```

A single-camera caller passes `&BaCameras::shared(&cam, quats.len())`:

```rust
let out = bundle_adjust(
    &BaCameras::shared(&cam, quats.len()),
    &mut quats, &mut trans, &mut points, &uv, &obs_img, &obs_pt,
    None, None, FreePointPolicy::default(), None, DEFAULT_PROTECTED_LOSS_SCALE,
    true, false, false, false, false, false, None, &DEFAULT_SCHEDULE, 60, 2, 12,
    &Progress::none(),
);
let solved_focal = out.cameras[0].focal_lengths().0;
```

**Why `BaCameras` rather than two more arguments.** The camera list and the
image-to-camera column only mean something together, and the checks on them
belong in one place: an empty camera list, an `image_camera` whose length is not
`n_img`, or an index past the list is a caller error and panics like the
kernel's other shape checks. `image_camera` is a `Cow` so that
`BaCameras::shared` can own the all-zero column it builds, while a caller with a
column of its own lends it (`image_camera: column.as_slice().into()`).

**Why `releases` narrows the flags rather than replacing them.** The six
flags are the kernel's whole-solve switches, and every caller that has one
decision for every camera keeps passing them as it always has, with `releases:
None`. A caller that decides camera by camera -- the reconstruction-level
adjustment, whose caller states a release per camera -- passes every flag on
and one `CameraRelease` per camera, and camera `j` releases its focal under
`opt_f && releases[j].focal`, its `k1` under `opt_k1 && releases[j].distortion`,
its spline under `opt_bspline && releases[j].distortion`, its principal point
under `opt_pp && releases[j].principal_point`, and its images' focal scales and
principal-point offsets under `opt_image_f && releases[j].image_focal` and
`opt_image_pp && releases[j].image_principal_point`, each still only where its
model admits it. With `releases: None` every camera takes the flags as they are
(`CameraRelease::ALL`). A list whose length is not `n_cam` is a caller error
and panics with the other shape checks.

**Why the result returns cameras.** A caller wants each solved camera back to
write into its own image set. The result carries the whole `CameraIntrinsics` per
input camera, in order: the input camera with its released parameters replaced
by the solved ones (the focal under `opt_f`, `k1` under `opt_k1`, the spline
under `opt_bspline`, where its model admits the release, the principal point
under `opt_pp`, and the shared value a per-image release re-centres), and equal
to the input camera otherwise. A caller copies it rather than reassembling it
from scalars.

`progress` is where the rounds and the LM iterations inside them are reported: a
phase per schedule round with an iteration count under it, so a caller knows
which round a long solve is in. It is also how the solve is asked to stop, which
it is between rounds and between iterations, those being the points where the
state is a whole answer rather than a half-written candidate. A stopped solve
returns the state it had reached rather than an error, since it has no `Result`
to put one in, and the caller asks `Progress::is_cancelled` again to find out.
`&Progress::none()` reports nothing and never stops: every method on it is a
branch on a null sink. Reporting changes nothing the solve computes: a run that
reports every phase into a sink returns the same output bit for bit as one
passed `&Progress::none()`, which the reconstruction-level adjustment's tests
assert ([../../gui/operation-progress.md](../../gui/operation-progress.md)).

Per schedule round:

1. **Retriangulate (rounds after the first).** Rebuild *every* point from
   *all* supplied observations at the current poses, through the point
   estimation operation
   ([triangulation-rules.md](../reconstruction/triangulation-rules.md)) with `marks`
   on for the round's direction mask, `few = absent`, and the floor, likelihood,
   cheirality and bar rules off (under `FreePointPolicy::cross`, `cheirality`
   on, and a free point starts the round from the best of three states, see
   "Re-estimation between rounds" under "The parametrisation"; a ranged or held
   point reads neither, see "Point constraints"). Each observation's ray is
   back-projected through the camera of its own image, so a track seen by two
   cameras reads each observation through its own lens. A track
   with fewer than 2 usable observations becomes `NaN`; a point with no
   observations at all becomes `NaN` too. What happens to such a point is the
   caller's choice: growth refills it from its full observation set after the
   adjustment, and the reconstruction-level adjustment deletes it.
   Re-admission is the point: observations a bad init lost re-enter once the
   refined cameras explain them.
2. **Trim.** Keep observations with residual norm `< trim_px`, an in-front
   measure over the round's in-front floor (below), and a finite point; then
   drop observations of points with fewer than `min_track` survivors. The
   trim never exceeds the invalid residual (`1e6` px), so an observation whose
   projection fails is left out whatever `trim_px` is, and `trim_px =
   f64::INFINITY` passes the residual test for exactly the observations that
   project; the
   between-round re-estimation's capped cost and its count of observations
   under the trim read the same bound. The
   in-front measure is model-aware: the canonical depth `−z_cam` for the
   perspective family, whose projection is defined only for `z_cam < 0`, and the
   range `‖p_cam‖` for a ray-path model (fisheye, equirectangular), which images
   rays out past `θ = 90°`; there the floor rejects only a point on the camera
   centre and the domain test is `ray_to_pixel`, whose failure is an invalid
   residual. A direction's measure is checked against zero (see "Points at
   infinity").
   If fewer than `min_obs` observations survive, return degenerate:
   `residual_norms` all `+∞`, `free_point_decision` `None`, and the poses,
   points and cameras as they stand at that moment. When the first round exits
   this way that is the caller's input; when a later round does, it is the poses
   and lenses the last completed round settled on, with the points this round
   re-triangulated from them. A wildly wrong focal is the usual cause, and a
   caller recognises the exit by its all-`+∞` residuals.
3. **Solve.** One robust sparse Levenberg–Marquardt solve (below) over the
   kept observations at the round's `loss_scale`.

After the last round, `residual_norms` is the unweighted reprojection
residual norm of **every supplied observation** at the final state (`+∞`
where invalid), so callers tally inlier fractions against denominators of
their own choosing.

### The in-front floor

A finite observation survives the trim when its in-front measure is greater
than

```
floor = 1e-6 · s,   s = median in-front measure over the round's
                        observations of finite points whose measure is positive
```

(`in_front_floor`, `IN_FRONT_FLOOR_FRACTION`), and a direction's observation
when its measure is greater than zero. With no finite observation in front at
all, the floor is zero.

**What the floor protects against.** The projection is singular at the camera
centre: the direction from the centre to a point there is undefined, and the
projection's derivative with respect to the point grows as one over the
point's range. The other place a perspective projection breaks, the image
plane (`z_cam → 0` at a finite range), needs no floor: a point there projects
far outside any image, and its residual is trimmed by `trim_px` before its
depth is read. Past the sign of the measure, which is the projection's own
domain test, the floor's one job is to reject a point at, or numerically at,
its camera's centre. Such a point comes from a triangulation that failed
(rays that cross at the camera that took one of them), and an observation of
it would hand that camera's translation a derivative orders of magnitude
larger than any other in the solve.

**Why a fraction of the scene.** Range is a length, and a reconstruction's
world unit is arbitrary: a solver's gauge, metres after a GPS fit, or anything
an `xform --scale` makes of it. A floor stated in any fixed unit trims a
different set of observations when the same scene is stated in another, so the
floor is a fraction of a length the scene states. The median in-front measure
is that length: it is positive whenever anything is in front, it is not moved
by a minority of far or near points, and, unlike the spread of the camera
centres, it does not vanish for a capture that rotates about one point. It is a
selection, not a sum, so it does not depend on the order of the observations.
It is read every round, at that round's state, from the same measures the trim
reads.

**Why `1e-6`.** A legitimately near point sits at a percent-level fraction of
the median: in a GLOMAP solve of `seoul_bull_sculpture`'s 17 images, whose
median depth is 1.44 world units, the nearest observation is at 8% of it before
the adjustment and at 6% after, and a quarter of the observations end within a
tenth of it. The fraction leaves four orders of magnitude between those and the
floor, and a point at the floor has a projection derivative a million times its
scene's typical one, which is the camera centre for every purpose the solve
has. The verdicts do not depend on the scale: scaling the world by any factor
scales the floor with it.

A floor stated through the focal compares pixels with world units and is not
scale invariant: `1e-3 · f` on that same solve (`f` = 341) would sit at 0.24
of the median depth, over 959 of its 2921 observations.

**Inverse depth.** A free point solved in inverse depth (`p̃ = ρ·p_cam`, see
"The parametrisation") is trimmed on the same measure, read from the position
the round handed back. In `(u, ρ)` the measure of the point is the measure of
`p̃` divided by `ρ`, so the test reads `measure(p̃) > 1e-6 · s · ρ`. A point
pulled onto its anchor (large `ρ`) meets the floor only where the anchor is
itself on a camera centre, which is the case of a track whose observing images
share one centre. As `ρ` goes to zero the bound goes to zero with it, and at `ρ = 0`
it is the direction's test, `measure(R_i · u) > 0`: the direction's floor of
zero is the same rule at zero inverse depth, not a separate one.

## The solve

Levenberg–Marquardt over a local parameterization, minimizing the soft-L1
robust cost applied per residual COMPONENT (scipy's element-wise
`loss="soft_l1"`)

```
cost = Σ_i s² · ρ(r_i² / s²),   ρ(z) = 2·(√(1 + z) − 1),   s = loss_scale
```

- **Parameters.** Per touched image a local `SO(3) × ℝ³` perturbation
  (`R ← exp(δθ)·R`, `t ← t + δt`); per touched point `X ← X + δX`; and per
  camera, when `opt_f`, its focal `f_j ← f_j + δf_j`; when `opt_k1`, its
  radial coefficient `k1_j ← k1_j + δk1_j`; when `opt_bspline`, its
  spline `c_j,i ← c_j,i + δc_j,i`; when `opt_pp`, its principal point
  `(cx_j, cy_j) ← (cx_j + δcx_j, cy_j + δcy_j)`. The four flags are requests
  to every camera, each decided on that camera's own model (below and "Several
  cameras"). Focal optimization requires a
  single-focal model whose projection multiplies `f` onto a distorted
  coordinate that does not itself depend on `f` — `SIMPLE_PINHOLE`
  (`x_d = rx/(−rz)`), `EQUIDISTANT_FISHEYE` (`x_d = θ·ûx` with
  `θ = atan2(ρ, rz)`), `SIMPLE_RADIAL_FISHEYE`
  (`x_d = θ·(1 + k1·θ²)·ûx` — `θ` comes from the ray, not from `r/f`,
  so the condition holds with distortion present), and the two spline
  models `SFMTOOL_FISHEYE` (`x_d = (θ + δ(θ))·ûx`) and `SFMTOOL_PINHOLE`
  (`x_d = (ρ + δ(ρ))·ûx` with `ρ = ρ_xy/rz`), whose coefficients are
  dimensionless and likewise ride on the ray's own radial coordinate. In
  all five `∂u/∂f = x_d = (u − cx)/f` at every incidence angle, the
  fisheye periphery past `θ = 90°` included. `opt_k1` is the fisheye family's radial
  distortion release and requires `SIMPLE_RADIAL_FISHEYE`:
  `∂(u, v)/∂k1 = f·θ³·(ûx, ûy)` exactly — the θ³ curvature the
  equidistant map cannot express, which is what lets the adjustment
  flatten a lens's residual field instead of absorbing it into the
  geometry (as a finite dome that pulls sky and horizon points in from
  infinity).
  Every other model fails the conditions — a second focal `fy` (no slot
  in the camera block), or higher polynomial coefficients recovered
  through `f`-dependent normalization. The binding rejects `opt_f` /
  `opt_k1` for those loudly, and the core silently holds that camera's
  parameter fixed (never a half-modeled DOF) while releasing it on the cameras
  whose models admit it. The `k1` release also needs the
  model's INVERSE to carry `k1`: retriangulation and direction
  re-estimation read `pixel_to_ray`, which for `SIMPLE_RADIAL_FISHEYE` is
  the Newton recovery of `θ` without the family's wide-angle blend — that
  blend hands back the identity `θ = r_d` ray past 90°, dropping `k1`
  exactly where `k1·θ³` is largest (a 105° rim at `k1 = 0.02` comes back
  6° off, which is a ray, not a rounding error).
- **The spline release.** `opt_bspline` is the radial distortion
  release of the two models that carry one, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`
  ([../../formats/sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md)),
  and only when that spline is defined: at least two coefficients on a
  positive finite domain end (`bspline_theta_max` / `bspline_rho_max`;
  anything shorter evaluates as the identity and has nothing to release).
  The released block is the camera's whole coefficient vector `c₀..c_{N−1}`,
  shared by every image that camera took like its `f` and `k1`, so with one
  camera the reduced camera system is `[f, k1, c₀..c_{N−1}, cx, cy | 6·n_im]`,
  width `4 + N + 6·n_im` (see "Several cameras" for more). `opt_k1` and
  `opt_bspline` are mutually exclusive on any one camera: no model carries both
  parameters, so the binding rejects the combination (checked before the model
  gates, so the caller sees the real reason) and the core degrades each on its
  own model test. Retriangulation and direction re-estimation read the model's Newton
  inverse, which carries the spline over the model's whole radial domain, for
  the same reason the `k1` release needs its own (for the fisheye that means no
  wide-angle blend; for the pinhole, an explicit Newton arm rather than the
  generic fixed-point undistortion).
  **The tail is regularized.** Observations thin out toward the image corners,
  so the outermost coefficients are often touched by one or two observations
  and would otherwise follow those few pixels' noise. A released spline
  `c = (c₀, …, c_{N−1})` of camera `j` therefore adds a second-difference
  penalty to the round's cost, the one the camera refit uses
  ([`../camera/refit-camera-intrinsics.md`](../camera/refit-camera-intrinsics.md)
  § "The span past θ_fit is regularized"):

  ```
  E_pen,j = w_j · Σ_{i ∈ rows_j} (f_ref,j · (c_{i−1} − 2cᵢ + c_{i+1}))²
  w_j     = SMOOTHING · n_rows,j / N,   n_rows,j = 2 · (camera j's kept observations in the round)
  ```

  `SMOOTHING = 1e-6` and the weight function `smoothing_weight` live beside the
  spline basis in
  [`bspline.rs`](../../../crates/sfmtool-core/src/camera/distortion/bspline.rs)
  and are shared with the refit, which counts two data rows per sampled ray
  where the adjustment counts two per kept observation; the penalty itself is
  `SplinePenalty` in
  [`bundle_adjust.rs`](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs).
  - *The focal in the row is fixed for the round.* `f_ref,j` is camera `j`'s
    focal at the start of the round, so the penalty is exactly quadratic in `c`
    and has no focal column. (The refit's unknowns are `f·cᵢ`, so its row is
    linear; with `f` and `c` separate, `f·Δ²c` would couple the penalty into
    the focal slot.)
  - *The gauge.* No row reaches the two gauge-anchored coefficients, which are
    implicitly zero: the first row is `c₀ − 2c₁ + c₂`. A row through them would
    pull `c₀` and `c₁` toward zero near the centre, where the data are densest
    and the gauge already fixes the value and slope. The last coefficient is
    held only by the last row, so with little data there it continues `c_{N−3}`
    and `c_{N−2}` in a straight line.
  - *The row set.* Every interior row, `i = 1, …, N − 2`, enters every round,
    whether or not data reach its coefficients. The set does not change with
    support, so the cost the damping ladder compares is one function for the
    whole round.
  - *In the normal equations.* With `P_j = √w_j · f_ref,j · D` for the round's
    rows, the penalty adds `P_jᵀP_j` to the camera block and `P_jᵀP_j·c` to its
    gradient, after the data and before the damping, so the Marquardt
    `λ·diag(h_cc)` reads the penalty too. The points do not enter it, so the
    Schur elimination is unchanged.
  - *In the cost.* Step acceptance compares the robust data cost plus
    `Σ_j E_pen,j` at the candidate's coefficients, and the relative improvement
    the termination test reads includes the penalty. The per-observation
    residuals the adjustment reports, `residual_norms`, are the data's alone.
  - *The weight is a decision for now.* `SMOOTHING` and the count of two rows
    per kept observation are the refit's, so the two agree on what a smooth
    tail is. The adjustment's data rows are soft-L1 with second-order scaling,
    so an observation's curvature is `(1 + z)^(−3/2)` of a plain row's, with
    `z = (r/s)²`, and at the last stage's loss scale of 1 px many inliers have
    `z` near or above one. The weight still counts each kept observation as
    two plain rows; counting each by its robust weight, or by `s²`, would
    change the penalty's share of the solve. On the `dino_dog_toy` case the
    weight keeps the default schedule's release off the monotonicity limit,
    but not a `[(12, 2)]` round run alone, which settles at the guard. On the
    synthetic tail fixture it lets the one observation in the last span pull
    the end slope to between 0.0024 and 0.42, on 11 of the 43 runs in which an
    observation reaches that span
    ([`bundle-adjustment-measurements.md`](bundle-adjustment-measurements.md#a-weakly-supported-spline-tail)).
    What a heavier weight would cost a well-supported tail is not measured,
    and the weight may be changed once there is more experience with real
    reconstructions.

  **The penalty places an unsupported coefficient.** A coefficient whose basis
  span no surviving observation touches has an exactly-zero data column, hence
  an exactly zero data diagonal (a sum of squares). It stays released, and its
  penalty rows alone place it: between supported coefficients it takes the
  smoothest curve through them (the fourth difference centred on it is zero, so
  a lone gap coefficient is the cubic through the two coefficients on each
  side), and past the last supported coefficient it continues the last two in
  a straight line (zero second difference), as the camera refit's span past
  `θ_fit` does. A coefficient that gains support within a round already has
  its rows, so nothing changes for it. Two cases are pinned, with the
  identity row and column and zero gradient entry that hold an unreleased
  `f` or `k1` and a frozen translation:
  - *A coefficient that loses its support within a round* keeps its value
    until the next round, whose first linearization decides afresh. Its
    value came from data in this round, and the loss comes from one of the
    round's own steps moving its observations out of its span before the
    poses have settled; placed by the penalty alone from there, it follows
    its neighbours through the unsettled steps that come next. On the kernel test in which far-field
    directions carry the spline, the rim observations leave the last
    coefficient's span after the first step; released, it followed a weakly
    supported neighbour to 0.12 (the truth is −0.07), every full step after
    that folded the spline, and the solve ended with the map 11.9 px off. On
    the folded-rim scene of the held-distortion retry the observations
    collapse until only `c₀` and `c₁` are supported; with the other six
    released, the candidate steps continued them from those two to values
    near 16, and the round ran out its iteration budget instead of settling
    at the guard.
  - *A spline with fewer than two supported coefficients* in a
    linearization is held there, every coefficient slot pinned: two are the
    fewest that anchor the straight line the penalty continues, and with
    one the rows would place the others relative to a single value the
    data hold. A camera with kept observations off the principal point has
    at least two supported coefficients, since every basis function active
    inside a knot span is positive there.

  The support test reads the diagonal from the data accumulation, before the
  penalty is added, since every row gives each of its coefficients a
  positive diagonal; support decides nothing else.

  Why every row, and why released: a coefficient released with the tiny
  column of an observation at the edge of its basis span and no row follows
  that observation without bound (0.0601 to 32,152 in one round on the tail
  fixture of the testing requirements), and one held at its input keeps
  whatever value an earlier model left there, which on that fixture leaves a
  wild starting end slope in place, between 0.0001 and 9.3, on 12 of 60 runs
  ([`bundle-adjustment-measurements.md`](bundle-adjustment-measurements.md#a-weakly-supported-spline-tail)).

  **Step guard.** A candidate spline is admissible when it is
  finite and keeps the model's monotonicity invariant `1 + δ'(d) > 0`
  everywhere on `[0, d_max]` and on the linear tail past it, where the single
  `1 + δ'(d_max) > 0` decides the whole half-line. The whole domain rather
  than the imaged field: monotonicity is the model's construction invariant
  (the bracket behind its inverse) and the accepted spline is persisted into
  the camera. Past the data the coefficients move as the penalty continues
  them, and a continuation of a falling slope can fold; the guard holds such
  a step, so the spline stops short of the continuation rather than folding.
  What the damping ladder does with a candidate that fails the guard is the
  next item.
- **The principal-point release.** `opt_pp` releases each camera's principal
  point, on every model. Every model projects as `u = fx·x_d + cx`,
  `v = fy·y_d + cy` with a distorted coordinate `(x_d, y_d)` read off the ray
  alone, so `∂(u, v)/∂(cx, cy)` is the identity, exact at every incidence angle
  and for a direction as for a finite point. It is not a scale, so it needs no
  focal release beside it, and on a two-focal model, whose focal cannot be
  released, it is released all the same. The camera's lens block carries `cx`
  and `cy` after its spline slots (`[f, k1, c₀..c_{N−1}, cx, cy]`), so the
  focal, `k1` and spline slots keep their places; unreleased, the two slots are
  pinned like an unreleased focal. The `k1` step guard's imaged field is
  measured about the principal point the round starts from.

  *The near-degeneracy.* A principal-point shift and a small rotation of every
  image of the camera move the image almost alike: in a pinhole camera a shift
  of `δ` pixels in `cx` and a rotation of `δ/f` about the camera's y axis differ
  only by a term of order `δ·x²/f²` at image coordinate `x`, plus whatever the
  distortion adds, since the distortion centre moves with the principal point
  and not with the rotation. So the principal point is determined by the
  field's width and by the distortion: well on a wide field or a strongly
  distorted lens, barely on a narrow undistorted one. The release has no prior.
  It is one pair of numbers fitted from every image of the camera, the
  Marquardt damping `λ·diag(JᵀJ)` keeps a step along a weak direction short,
  and the step guard below keeps it on the sensor. On a 60° field a principal
  point planted 2% of the width off centre is recovered to within 0.05 px from
  noise-free observations; on a 5° field the solve converges with the principal
  point on the image and a residual no higher than with it held, and recovers
  nothing in particular, which the data cannot support. The kernel reports no
  conditioning for it; the reconstruction-level adjustment reports how far it
  moved.
- **Step guards and the held-distortion retry.** Inside the damping ladder
  each camera's candidate lens is checked. A focal that is not finite and
  above `1e-6`, a principal point that is not finite or lies off the image
  (`cx` outside `[0, width]` or `cy` outside `[0, height]`; no physical lens
  has its optical axis off the sensor, and a solve heading there is following
  a degenerate direction rather than the data), or a `k1` or spline
  coefficient that is not finite, rejects the whole step, which is
  re-damped. A finite `k1` that folds the imaged field (it
  fails `k1_step_admissible`) or a finite spline that breaks monotonicity does
  not: the same rung solves the step again at the same `λ` with that camera's
  distortion slots (the `k1` slot, or every spline slot) pinned in the reduced
  system, with the identity row and column and zero gradient entry the support
  pinning uses. Pinning a slot after the Schur elimination is the same as
  removing that variable from the damped system, so the back-substituted
  points are the exact solution with the distortion held, and the focal, the
  principal point, the poses, the points and every other camera's lens still
  move. The held
  candidate is checked again: a held spline is the current, admissible one,
  but a held `k1` is re-checked because `k1_step_admissible` reads the
  candidate focal, and its failure rejects the whole step; a camera whose
  distortion the held step makes inadmissible joins the held set and the step
  is solved again, which ends within `n_cam` solves since the set only grows.
  A held step is accepted or rejected on the cost like any other, and the next
  rung, or the next iteration, tries the full step first again. The reduced
  system is copied before its factorization for this, in every rung of a
  solve that releases a distortion. The `λ > 10¹²` cap ends the ladder after
  every kind of rejection.

  Why: rejecting the whole joint step because the distortion part crosses
  the guard raises `λ` for every slot under Marquardt scaling, so the poses'
  and points' steps shrink with the distortion's, and two such steps end the
  round by the tiny-step test with the poses and points unsettled. The held
  step is not the constrained optimum along the guard. It moves everything
  except the distortion, and it needs no constrained solver. Its limit: when
  the full step crosses the guard because it overshoots rather than because
  the optimum lies past the limit, the held step is accepted at the same `λ`,
  `λ` falls, and a damped full step that would have moved the distortion
  part of the way is not tried, so the distortion can stay where it was for
  the round (measured in
  [`bundle-adjustment-measurements.md`](bundle-adjustment-measurements.md#a-weakly-supported-spline-tail);
  a damped distortion step in its place is an
  [open question](#open-questions)).
- **Staged releases.** Callers open the shared parameters in stages:
  fixed → `opt_f` → `opt_f` plus a distortion release, so a distortion
  parameter opens only on a focal that has already settled — `opt_f + opt_k1`
  for the `k1` release, `opt_f + opt_bspline` for the spline. The spline
  **must** be co-released with the focal, even where an earlier stage froze
  `f`: under the center-anchored gauge the spline pins `δ(0) = δ'(0) = 0` and
  so cannot express a central-scale correction at all — that is what `f` is
  for — so a spline released against a frozen focal could only bend the
  periphery around a scale it has no way to fix. A caller honoring an earlier
  focal decision therefore guards the released *map* rather than the raw `f`
  (the equivalent equidistant focal of the composite map), refitting with `f`
  frozen only when that guard trips.
  The reconstruction-level adjustment
  ([`../reconstruction/bundle-adjust.md`](../reconstruction/bundle-adjust.md))
  exposes this pair per camera, as a `CameraRelease` of `focal` and
  `distortion` for each camera, which it passes as `BaCameras::releases` with
  every flag on, so each camera frees whichever of `k1` and the spline its model
  has, and refuses the distortion without the focal, shared or per image (a
  per-image focal moves the camera's focal to its images' mean). The same
  `CameraRelease` carries `principal_point` and the per-image releases, which
  stand apart from this staging. That is how the
  viewer's Bundle Adjust dialog, the MCP `bundle_adjust` tool and `sfm xform
  --bundle-adjust` on a spline camera reach it.
- **Jacobian.** The projection block `∂(u, v)/∂p_cam` — analytic from
  `CameraIntrinsics::ray_to_pixel_with_jacobian` for the perspective
  family (`SFMTOOL_PINHOLE` included, whose radial spline enters the
  family's own `x_d = x·g(r²)` form as `g(ρ) = 1 + δ(ρ)/ρ`),
  `EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE` (the chain
  through `θ_d = θ·(1 + k1·θ²)` is closed-form) and `SFMTOOL_FISHEYE`
  (the same chain at `θ_d = θ + δ(θ)`), a central difference of
  `ray_to_pixel` for the remaining polynomial fisheye models and
  equirectangular, which have no analytic form, with a step of `1e-6` of the
  camera-frame point's range (the projection is homogeneous of degree zero, so
  its derivative scales as one over the range and a step that scales with it
  is equally accurate at every world scale) — composed with
  `−[R·X]ₓ` (rotation), `I₃` (translation), and `R` (point) blocks,
  exactly as in `pose_refine.rs` (including the fallback). An observation
  whose point is behind the camera / outside the model domain contributes
  residual `(1e6, 0)` with a zero Jacobian row — penalized, never
  steering. The shared-parameter columns are analytic too: `(u − cx)/f`
  for the focal, the identity for the principal point, `f·θ³·(ûx, ûy)` for
  `k1`, and
  `∂(u, v)/∂cᵢ = f·Bᵢ(d)·(ûx, ûy)` for spline coefficient `i` over the
  model's radial coordinate `d` (`θ` for `SFMTOOL_FISHEYE`, `ρ = ρ_xy/rz`
  for `SFMTOOL_PINHOLE`), all with `d` and `û` read from the ray in the
  optical frame, so all are exact over the whole field, the fisheye's
  periphery past 90° included; on the spline's linear tail past `d_max` the
  correction continues along its end tangent, so the column is the exact
  `f·(Bᵢ(d_max) + B'ᵢ(d_max)·(d − d_max))·(ûx, ûy)`, both terms read from the
  one clamped basis evaluation. A direction (point at infinity) takes every one of these
  columns unchanged — it projects through the very same map, at `R·d`
  instead of `R·X + t`. Cubic local support means at most four spline
  columns are non-zero at one observation, so the per-observation camera
  block stays fixed-width (`[f, k1, 4 active coefficients, (cx, cy,) 6 pose]`)
  however long the spline is; the observation's own knot span decides
  which coefficient slots those four columns scatter into, and an active
  basis function from the gauge-anchored pair has no coefficient and no
  column.
- **Robust weighting.** Second-order (Triggs-style) scaling, exactly
  scipy's `scale_for_robust_loss_function`: per residual component with
  `z = (r/s)²`, the Jacobian row scales by `√(ρ' + 2ρ''z)` — for soft-L1
  `(1 + z)^(−¾)` — and the residual by `ρ'/√(ρ' + 2ρ''z) = (1 + z)^(+¼)`,
  so `Jᵀr` is the true robust gradient while `JᵀJ` carries the corrected
  curvature. The true robust cost (not the surrogate) decides step
  acceptance. First-order IRLS is not used: its curvature model is
  shallower, and on the seoul bull images it stopped a released focal short,
  at its starting value, where the second-order scaling moves it the 20% to
  the reference focal.
- **Schur complement.** Points are eliminated: per-point 3×3 blocks are
  inverted directly and the reduced camera system (one lens block
  `[f_j?, k1_j?, c_j,0..c_j,N_j−1?, cx_j?, cy_j?]` per camera, then `6·n_im`,
  dense) is
  solved by LU; point updates back-substitute. Unreleased lens slots (and a
  released spline's coefficient slots that lost their support within the
  round, every coefficient slot of a released spline with fewer than two
  supported coefficients, and every slot of a camera no kept observation
  reaches) are
  pinned to an identity row/column with a zero gradient entry, which is what
  keeps that system regular.
  Rejected steps re-damp and re-solve from the same linearization (no
  re-evaluation), with Marquardt scaling `λ·diag(JᵀJ)` for the
  `x_scale="jac"` parameter-scale invariance of the scipy original.
- **Termination.** `max_iters` accepted-step budget per round. Each accepted
  step is a **full** step or a **held** step (one retried with a distortion
  held), and the round ends in one of four ways:
  - *Converged*: two accepted full steps in a row each improve the cost by
    less than `1e-8` relative (a single small improvement is also how a long
    run of small steps across a nearly flat stretch of the cost begins, and a
    focal release can move the focal 20% across one). An accepted held step
    ends the run.
  - *Settled at the guard*: two accepted held steps in a row each improve the
    cost by less than `1e-8` relative; an accepted full step ends the run. The
    poses and points have settled with the distortion held, so the round
    stops, but the step that would lower the cost further is one the guard
    forbids, and the round does not count as converged.
  - *Ladder exhausted*: no rung of a bounded ladder (12 ×4 escalations,
    capped at `λ = 10¹²`) finds a downhill step. The round ends settled at the
    guard if any rung of that ladder made a held retry, and converged
    otherwise.
  - *Iteration budget*: `max_iters` accepted steps, or a cancellation; not
    converged.

  A round that ends settled at the guard says so on `progress`:
  `the round settled with a distortion held at its step guard: camera 0
  (spline end slope 0.013)`, naming each camera the latest retry held, with
  its end slope `1 + δ'(d_max)` for a spline and its value for a `k1`
  (`camera 1 (k1 -0.1234)`). Only the final round's ending reaches
  `FreePointDecision::converged`, which is true for *converged* alone.

## Several cameras

Everything above is stated for one camera, and a solve over several is the same
solve with the lens parameters kept per camera. `BaCameras` carries the camera
list and, per image, the index of the camera that took it.

### The reduced camera system

The reduced system has one lens block per camera, then the poses:

```
[ f₀?, k1₀?, c₀,₀..c₀,N₀−1?, cx₀?, cy₀? | f₁?, k1₁?, c₁,₀..c₁,N₁−1?, cx₁?, cy₁? | … | 6·n_im ]
```

Camera `j`'s block has a focal slot, a `k1` slot, `N_j` spline slots and the
principal point's two slots, where `N_j` is the length of its spline when its
spline is released and `0` otherwise. The principal point comes after the
spline so that the focal, `k1` and first spline slot keep fixed offsets
(`F_SLOT`, `K1_SLOT`, `BSPLINE_SLOT0`) within the block. The system's width is
`Σ_j (4 + N_j) + 6·n_im`. An observation of image `i`
writes its lens columns into the block of camera `image_camera[i]` and into no
other camera's block, so two cameras' lens parameters couple only through the
poses and points they both observe. A few cameras add a few slots beside the
`6·n_im` pose slots, so the dense solve costs what it does with one.

The per-observation camera block keeps a fixed width, one of six, each a
compile-time instantiation of the solve: `2 + 6` columns when no camera
releases a spline or a principal point, `2 + 4 + 6` when some camera releases a
spline, each of those with the principal point's two columns before the pose
(`4 + 6`, `4 + 4 + 6`) when some camera releases its principal point, and each
of those two with the image's three adjustment columns after the pose
(`4 + 6 + 3`, `4 + 4 + 6 + 3`) when some camera releases a per-image parameter
(see "The per-image release"), where each image's block in the reduced system
is nine wide rather than six. In
the spline instantiations a spline column that carries no coefficient (one of
the gauge-anchored pair, or any spline column of a camera whose spline is not
released) points at the observing camera's own `k1` slot and is exactly zero,
so it adds exact zeros there whether or not that slot is released. In the
principal-point instantiations a camera whose principal point is not released
has zero columns there, pointing at its own pinned `cx`, `cy` slots.

The principal point's columns are in the block only when a principal point is
released, rather than always, because the Schur elimination's cost grows with
the square of the block's width over every pair of observations of every
point: carrying them in every solve makes the kernel's own tests about 40%
slower, solves that release no principal point included. A solve that releases
none computes exactly what it would if the lens blocks had no `cx`, `cy` slots
at all: those slots are pinned identity rows that no observation reaches, and
they change no other entry of the reduced system or of its solution.

Every rule the single block follows applies to each block separately:

- **Release gates, per camera.** `opt_f`, `opt_k1`, `opt_bspline` and
  `opt_pp` are requests to every camera, narrowed per camera by
  `BaCameras::releases` where it is given. A camera whose model the release is not exact for keeps that
  parameter fixed, so a `SIMPLE_PINHOLE` and an `OPENCV_FISHEYE` in one solve
  under `opt_f` release the pinhole's focal and hold the fisheye's, and a camera
  whose entry releases nothing is held whatever the flags say.
- **Pinning.** A slot that is not released is pinned with an identity row and
  column and a zero gradient entry, and so is every slot of a camera none of
  whose images has a kept observation in the round. Such a camera is not
  stepped, and comes back exactly as it went in.
- **Step guards.** A candidate step is rejected, and re-damped, when any
  camera's candidate focal is not finite and above `1e-6`, its principal point
  is not finite or lies off its own image, its distortion step is not finite,
  or an image's focal or adjusted principal point fails the per-image guards
  ("The per-image release"). A camera whose `k1` folds inside its own imaged field
  (the largest pixel radius of its own kept observations about its own
  principal point), or whose spline breaks monotonicity on its own domain, has
  its distortion held for a retry of the step, and only that camera's: the
  other cameras' distortions move in the held step.
- **Staged releases** are the caller's schedule, as with one camera.

### What reads the camera, per observation

Everything that reads a camera reads the intrinsics of the observation's image:
the camera `cameras[image_camera[obs_img[k]]]` with the image's adjustment
applied, which is the camera itself for an image with none. That is the
projection, its Jacobian and the lens columns; the trim's in-front measure; and
`pixel_to_ray` in the inter-round re-estimation. The in-front floor is the
scene's, not a camera's, so it reads no camera.

Under `FreePointPolicy::cross` the storage decision reads each camera twice
more. The noise level pools every camera's residuals and gates each camera's
against its own robust spread, as the stored measure does, each residual read
through its image's intrinsics and gated with its camera's, and each ray's noise
weight is the projection derivative of its own image's intrinsics, so a track
seen through two lenses weights each sighting by the pixels per radian of the
lens that took it. The noise level's floor is the cameras' keypoint resolution.

### Held image adjustments

`BaCameras::image_adjustments` gives each image an intrinsic adjustment of its
camera, a focal scale `s_i` and a principal-point offset `a_i` in the short-side
unit `m = min(width, height)`, so that image `i` reads its camera at
`f_i = f·(1 + s_i)` (both focals on a two-focal model) and
`(cx_i, cy_i) = (cx, cy) + a_i·m`
([`IntrinsicAdjustment`](../../../crates/sfmtool-core/src/camera/intrinsics.rs)).
Unless a camera releases them per image ("The per-image release", below), the
solve holds them: they are inputs, read wherever a camera is read (above). The released parameters are still the
camera's shared ones, with each image's adjustment applied on top of them, so a
camera's lens block is unchanged and its columns from an adjusted image are:

```
∂(u, v)/∂f       = (1 + s_i)·(x_d, y_d) = (u − cx_i, v − cy_i) / f
∂(u, v)/∂k1      = f_i·θ³·û
∂(u, v)/∂cⱼ      = f_i·(Bⱼ(d) + B'ⱼ(d)·(d − d_max)₊)·û
∂(u, v)/∂(cx,cy) = I
```

the unadjusted columns with the image's focal in place of `f` and the focal
column read about the image's principal point. The focal column keeps `f` in
its denominator because the parameter is the shared focal, which the image's
focal is `1 + s_i` times. The `k1` step guard's imaged field reads each
observation's pixel radius about its image's principal point divided by
`1 + s_i`, since the image's map folds at `1 + s_i` times the shared map's
radius. Where the camera releases its principal point, the step guard checks
each image's adjusted principal point as well as the shared one.

An image with an adjustment is linearised through an entry of its own, after
the cameras', whose camera is the adjusted one and whose lens slots are its
camera's, and the robust cost the damping ladder compares reads the same
entries. Those entries are built per image per evaluation, never per
observation. `image_adjustments` of `None`, or of all-zero adjustments, adds no
entry and reads every camera as it is, so the solve is the one without
adjustments bit for bit. A length other than `n_img` is a caller error and
panics with the other shape checks.

### The per-image release

`opt_image_f` and `opt_image_pp` release each image's intrinsic adjustment:
its focal scale `s_i` on the models `opt_f` admits (`focal_is_releasable`), and
its principal-point offset `a_i` on every model, narrowed per camera by
`CameraRelease::image_focal` and `CameraRelease::image_principal_point`. They
are for a lens whose focal and principal point move a little from one
photograph to the next: a phone camera's autofocus changes its focal with the
focus distance, and its optical image stabilisation shifts the lens sideways,
which moves the principal point
([`bundle-adjustment-measurements.md`](bundle-adjustment-measurements.md#per-image-intrinsic-adjustments)).
The distortion stays the camera's, being the part the images share and the part
that needs observations from many images to reach the edge of the field.

**Per-image values only.** A camera that releases a per-image parameter holds
its shared value of the same kind for the solve: `image_focal` holds the shared
focal and `image_principal_point` the shared principal point, whatever `opt_f`,
`opt_pp` and the camera's `focal` and `principal_point` say. The data depend on
a shared value and an image's adjustment only through their sum, so releasing
both would leave a direction no observation sees. Holding the shared value
removes it; the prior below acts only on each image's deviation from the mean
of the camera's images, so the data alone decide that mean; and after every
round the camera takes the mean (below), which is how the shared value moves.
Asking for the shared and the per-image release of one parameter is therefore
the per-image release alone, bit for bit. A per-image focal stands for the focal
a spline release needs: the images' mean scale is the central scale the spline
cannot express.

**The log focal scale.** The focal is solved, held and re-centred in log space:
the parameter is `t_i = ln(1 + s_i)`, so the image's focal is `f·exp(t_i)`. A
focal scale is multiplicative, and in `t` a focal `k` times the camera's and
one `1/k` times it lie the same distance from it, `±ln k`, where in `s` they
lie `k − 1` and `1 − 1/k` away; the prior, the mean it centres on, the
re-centring and the widths below all read `t`, so they treat the two alike.
Storage is unchanged: the file, `IntrinsicAdjustment::focal`, the kernel's
`image_adjustments` in and out and every accessor hold `s`, with
`f_i = f·(1 + s_i)`. The kernel converts where it reads and writes the
parameter, `t = ln(1 + s)` (`s.ln_1p()`) in and `s = exp(t) − 1`
(`t.exp_m1()`) out. For small scales `t ≈ s − s²/2`, so `t` reads as a fraction
of the focal as `s` does and a width keeps its meaning to first order,
`σ_t ≈ σ_s`; the mean of `t` and the arithmetic mean of `s` differ by about
`var(s)/2`, which for the 1% spread of a phone's autofocus is `5·10⁻⁵`.

**The columns.** Each image's block of the reduced system grows to
`[δθ×3, δt×3, t_i, a_x, a_y]`, nine wide, in a solve where some camera releases a
per-image parameter, and stays six wide otherwise; an image's slots its camera
does not release are pinned. The per-observation camera block carries the same
three columns after the pose in two more instantiations, `BASE_IMG_CAM_COLS`
and `BSPLINE_IMG_CAM_COLS`, which also carry the shared principal point's two
columns, so the widths stay one instantiation each and number six. At the
image's adjusted camera:

```
∂(u, v)/∂t_i   = (u − cx_i, v − cy_i)     (= f·exp(t_i)·(x_d, y_d))
∂(u, v)/∂a_x,i = (m, 0)
∂(u, v)/∂a_y,i = (0, m)
```

exact on every model the release admits, as the shared focal's and principal
point's are: the model projects `u = f_i·x_d + cx_i` with `x_d` independent of
the focal, and `∂f_i/∂t_i = f_i`, so the focal column is the pixel's offset from
the image's own principal point. An image of a camera that releases a
per-image parameter is read through a lens entry of its own whatever its
adjustment, since the adjustment moves within a round.

**The prior.** Each camera `c` holds its images' deviations from their mean with
a Gaussian prior of widths `σ_t,c` (the log focal scale's, read as a fraction)
and `σ_pp,c` (the principal point's, in the short-side unit, the same on both
axes):

```
σ_n² · Σ_i ( ‖a_i − ā‖² / σ_pp,c²  +  (t_i − t̄)² / σ_t,c² )
```

over the camera's images the round solves (its posed images with a kept
observation), with `ā` and `t̄` their means. `σ_n` is the reprojection noise
level at the state the round starts from, over its kept observations, measured
as the storage decision measures it (`round_noise`), or the round's loss scale
where it measures nothing; it is fixed for the round, so the cost the damping
ladder compares is one function per round, as the spline penalty's `f_ref` is.
The prior is in the data's units, `Σ r²` in pixels, and the offsets and their
width share a unit, so `m` cancels. A camera with one image in the round has no
deviation to hold, and that image is solved by its data alone.

The prior couples the images of a camera. With `w = σ_n²/σ²` and `n` images it
adds `w·(I − 𝟏𝟏ᵀ/n)` to the camera's per-image slots of one parameter and
`w·(x_i − x̄)` to each image's gradient entry, after the data and before the
damping, and its value to the cost the ladder compares. It is added exactly,
into the dense reduced system: that system is already dense and factored by
LU, so the `n²` entries per camera and parameter add no structure the solve
must respect and cost nothing beside the `(9·n_im)²` it holds, and the prior
reads no point, so the Schur elimination is unchanged. A camera-level mean as a
slot of its own, with each image's prior centred on it, reaches the same
minimum once that slot is eliminated, but adds slots, a mean to pin for a camera
of one image, and a sparse block the dense factorization cannot use.

**Deriving the widths.** The widths are derived for each camera from the solve,
an empirical Bayes estimate, because the right width differs between cameras (a
stabilised phone lens moves its principal point by a percent of the image, a
fixed lens by nothing) and a width read off a clean reconstruction holds an
ordinary one too tightly:

1. *A loose first round.* The first round runs every prior at
   `ImagePriorOptions::loose`, wide enough that a deviation the data support is
   pulled very little; its job is to stop a runaway, an image with few
   observations whose principal point would otherwise follow the rotation it is
   nearly degenerate with off the image.
2. *Each image's variance.* After a round, each image's data-only block `H_i` of
   the reduced system is read at the round's last linearisation: its pose and
   adjustment slots, the points eliminated undamped, before the prior and the
   damping (a ranged point's reference columns, a share of another image's
   observation, are left out). Inverting the whole block over the slots that
   move counts the adjustment's coupling with the same image's pose against it,
   which is the near-degeneracy the prior exists for. `V_i = σ_e²·[H_i⁻¹]_aa` is
   then the adjustment's covariance with every other image and the shared
   camera held, where `σ_e` is the noise level of the round's solved residuals,
   read at the state the round ended at: the level it started from still holds
   the start's error, which in a first round from an unadjusted camera can be
   ten times the noise. An image whose block is not positive definite gives no
   sample.
3. *Undoing the prior's pull.* The solved deviation `d_i = a_i − ā` was pulled
   toward the mean by the round's prior, whose curvature on the image's own
   block is `W = σ_n²·P⁻¹` (`P = diag(σ_t², σ_pp², σ_pp²)`, the round's widths).
   The deviation the data alone give is `d̂_i = (I + H_i⁻¹·W)·d_i`, every
   released parameter together, since one image's focal scale and principal
   point can be correlated. This is exact for the image's own block with every
   other image held; the prior's own diagonal there is `w·(1 − 1/n)` rather than
   `w`, a factor left out, which makes `d̂_i` slightly small for a camera of few
   images. Without the undoing, a width derived from pulled deviations and
   solved under again would shrink the next spread, running the width down to
   the floor wherever the images' variances are comparable to the spread.
4. *The spread.* `d̂_i` is the true deviation plus noise of variance `v_i`, so
   its variance is `τ² + v_i` for a spread `τ`. The camera's spread `s` solves
   `s² = max(0, m² − median_w v)` with `m = 1.4826·median_w |d̂ − c|`, every
   median weighted by `1/(s² + v_i)`, each sample's precision as a sample of the
   spread, and iterated from the round's width; `c` is each axis's weighted
   median of `d̂`. The principal point pools its two axes as two samples per
   image. Medians, so one wild image counts as one sample; the weight, so an
   image whose noise swamps the spread counts little while images whose noise is
   well under it count alike; the centre a median rather than the mean, so an
   image that moves the mean by its share, which moves every deviation alike,
   does not widen the spread. `1.4826` makes the median absolute deviation of a
   Gaussian its standard deviation. What remains errs high: `d̂_i` carries the
   other images' errors as well as its own, so its noise is larger than the
   conditional `v_i` the subtraction removes.
5. *The width.* The camera's next width is `s` clamped to
   `[ImagePriorOptions::floor, ImagePriorOptions::loose]`: the floor keeps a
   camera whose images do not vary from being pinned to its mean by a spread
   that rounds to nothing, and the loose width as a cap keeps a derived width
   from releasing more than the first round did. A parameter with fewer than
   two images giving a sample keeps the loose width and has no spread.

Every step reads the focal as the log focal scale `t_i`: the deviation is
`t_i − t̄`, the variance is that of `t_i`, since the solve's column is `t`'s, and
the spread, the floor and the loose width are of `t`. `ImagePriorOptions` keeps
its defaults, 0.1 loose and 0.001 floor, which read as the same fractions of the
focal they would of `s`.

The widths are derived again before every round after the first, from the round
just finished, and the round runs under them: the rounds are where the cost is
already fixed and re-measured, and re-deriving follows the trim, so the final
round runs under a width measured on the observations the earlier rounds left.
The previous width enters a derivation only through the pull the undoing
removes, so deriving again on the same observations gives the same width up to
the linearisation, rather than ratcheting it down. A schedule of one round runs
that round a second time under widths derived from the first, so every derived
prior is derived at least once; with `ImagePriorOptions::fixed` every round runs
under the fixed widths as given, without the floor or the cap, and nothing is
derived.

**Re-centring after every round.** After each round, every camera with a
per-image release takes the mean of the released adjustments over its images
the round solved, and every image of the camera keeps its difference from it:
`cx += mean(a_x)·m`, `cy += mean(a_y)·m` and `a_i −= mean(a)` for the principal
point; `f ← f·exp(t̄)` (every focal of a two-focal model) and `t_i −= t̄` for
the focal, with `t̄` the mean of the log focal scales, so that the mean of `t`
is zero to rounding. In `s` that is `f ← f·g` and `1 + s_i ← (1 + s_i)/g` with
`g` the geometric mean of `1 + s_i`, which differs from the arithmetic mean by
about `var(s)/2`. It is applied to every image of the camera in the solve's
arrays, the ones the round did not solve included, so every image's intrinsics
are unchanged by it, to rounding. The mean is over the images the round solved, the set the
prior's mean is over, because those are the images whose adjustments the data
fixed. It runs after every round rather than once at the end so that the shared
value is the images' mean at the start of every round: the spline penalty's
`f_ref`, the `k1` guard's imaged field and the shared principal point's guard
read the shared value, and each reads it as the camera's typical image only when
the adjustments are centred on it. The prior and the derivation measure about
the mean, so the re-centring changes neither.

**Step guards.** A candidate step is rejected and re-damped when an image's
log focal scale `t_i` is not finite, its focal `f·(1 + s_i)` with
`s_i = exp(t_i) − 1` is not finite and above `1e-6`
(`image_focal_step_admissible`), or its adjusted principal point is not finite
or leaves the image. `1 + s_i = exp(t_i)` is then positive by construction. The
principal point's test reads every image of a camera that releases its
principal point, shared or per image, whose adjustment is released or not zero;
an image at a zero held adjustment is the shared principal point, which its own
guard checks.

**The result.** `BundleAdjustment::image_adjustments` holds every image's
adjustment after the solve, and `image_priors` one `Option<ImagePrior>` per
camera: `widths`, the widths the final round ran under (`NaN` for a parameter
the camera does not release per image), `derived`, false under `fixed`, and
`spread`, the last derivation's spread before it was clamped (`NaN` for a
parameter not released per image or with fewer than two samples, and both `NaN`
when nothing was derived). The camera in `cameras` carries the re-centred shared
values.

**Parity.** A solve in which no camera releases a per-image parameter takes the
six-wide image blocks and the instantiations it took before the release existed,
reads all-zero adjustments as none, and computes the same output bit for bit:
the kernel's, the reconstruction-level adjustment's, growth's and rotation
initialisation's test scenes (377 solves) gave identical outputs before and
after the release was added.

### Parity

With one camera and every image on it, the layout above reduces to
`[f, k1, c₀..c_{N−1}, cx, cy | 6·n_im]` and every per-camera read reads that camera, so
the solve is the one the sections before "Several cameras" describe, bit for bit
on every output: poses, points, residual norms, representation and the returned
camera.

## Bindings

```python
bundle_adjust(
    cameras,                   # sequence of CameraIntrinsics, one per camera
                               # (each carries its own initial f)
    image_camera,              # (n_img,) uint32 index into `cameras`
    quaternions_wxyz,          # (n_img, 4) world-to-camera (WXYZ)
    translations,              # (n_img, 3)
    points,                    # (n_pt, 3), NaN allowed
    uv,                        # (n_obs, 2)
    obs_image,                 # (n_obs,) uint32
    obs_point,                 # (n_obs,) uint32
    point_at_infinity=None,    # (n_pt,) bool; a marked row of `points` is a
                               # world-frame direction, the representation the
                               # point starts in. None and all-False agree bit
                               # for bit
    held=None,                 # (n_pt,) bool; a held point's coordinate is the
                               # caller's for the whole solve
    distance=None,             # (n_pt,) float64; a ranged point's distance,
                               # +inf for a direction, NaN where not ranged
    distance_from=None,        # (n_pt,) image index, or a sequence of image
                               # indices to average, with -1 where there is
                               # none; a finite `distance` requires one
    free_points_cross=True,    # solve free points in inverse depth and decide
                               # each one's representation at the end, on the
                               # point-or-bearing test at the measured noise;
                               # False keeps the caller's representation
    protected=None,            # (n_obs,) bool; protected observations survive
                               # every trim gate and take the wider loss scale.
                               # None/all-False reproduces the unprotected
                               # behavior bit for bit
    protected_loss_scale=3.0,  # multiplier on each stage's loss scale for
                               # protected observations (positive and finite)
    opt_f=False,               # every camera SIMPLE_PINHOLE,
                               # EQUIDISTANT_FISHEYE, SIMPLE_RADIAL_FISHEYE,
                               # SFMTOOL_FISHEYE or SFMTOOL_PINHOLE
    opt_k1=False,              # every camera SIMPLE_RADIAL_FISHEYE
    opt_bspline=False,         # every camera SFMTOOL_FISHEYE or
                               # SFMTOOL_PINHOLE with a defined spline;
                               # exclusive with opt_k1
    opt_pp=False,              # any camera model
    opt_image_f=False,         # each image's focal scale, on the opt_f models
    opt_image_pp=False,        # each image's principal-point offset, any model
    image_adjustments=None,    # (n_img, 3) [focal, a_x, a_y]; None = none
    image_prior=None,          # {"loose", "floor", "fixed"}: dicts of focal
                               # and principal_point; None = the defaults
    schedule=[(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)],
    max_iters=60,
    min_track=2,
    min_obs=12,
) -> dict                      # cameras (list of CameraIntrinsics),
                               # quaternions_wxyz (n_img, 4),
                               # translations (n_img, 3), points (n_pt, 3),
                               # residual_norms (n_obs,),
                               # point_at_infinity (n_pt,),
                               # free_point_decision (dict or None),
                               # image_adjustments (n_img, 3),
                               # image_priors (list of dict or None)
```

`cameras` in the result holds one `CameraIntrinsics` per input camera, in
order: the kernel's returned cameras, each the input camera with its released
parameters replaced by the solved ones. A caller reads a solved focal as
`out["cameras"][j].focal_lengths[0]` and a solved `k1`, spline or principal
point from the camera's `parameters`.

The binding holds every camera to every release it is asked for: a release that
some camera's model does not admit raises a `ValueError` naming that camera,
rather than letting the kernel hold that camera's parameter fixed. `opt_bspline`
raises for a camera that is neither `SFMTOOL_FISHEYE` nor `SFMTOOL_PINHOLE`, and
for an undefined spline; `opt_k1` together with `opt_bspline` raises first, so
the caller sees the exclusion rather than whichever model gate happens to fire.
An empty `cameras`, an `image_camera` whose length is not `n_img`, and an index
past `cameras` raise `ValueError` as well. `opt_image_f` raises for a camera
whose focal `opt_f` would refuse. `image_adjustments` is the kernel's
`BaCameras::image_adjustments`, one `[focal, a_x, a_y]` row per image, held
unless a per-image release moves it, and `image_adjustments` in the result is
`BundleAdjustment::image_adjustments` in the same layout. `image_prior` is
`ImagePriorOptions` as a dict of `loose`, `floor` and `fixed`, each a dict of
`focal` and `principal_point` (`fixed` also `None`), a missing key at its
default; an unknown key, or a width that is not positive and finite, raises.
`image_priors` holds one entry per camera, `None` where it released nothing per
image, and otherwise a dict of `widths`, `derived` and `spread`.

`held`, `distance` and `distance_from` are the flat form of the kernel's
`PointConstraints`: the binding assembles them, so a caller states each point's
constraint in the same array layout it states the rest of its per-point data. A
row is ranged where `distance` is not `NaN`; `distance_from` naming a single
index is `DistanceReference::Image` and one naming a sequence is
`DistanceReference::ImageMean`.
The assembly is `PointConstraints::from_arrays`, in the kernel rather than in the
binding, because a Rust caller reading the same three statements off a
reconstruction's constraint columns
([`../reconstruction/bundle-adjust.md`](../reconstruction/bundle-adjust.md))
has to reach the same verdicts; the binding maps its refusals onto `ValueError`
and adds none of its own. `None` comes back where every point is free, which is
the off position the parity requirement is stated against.
The refusals are a point that is both held and ranged, a distance that is
not strictly positive, a finite distance with no origin, and an origin past the
image set.
`distance_from` is ignored on a `NaN` or `+inf` row, which is measured from
nothing.

`point_at_infinity` in the result is the representation each point ended with,
and is how a caller learns the outcome of the storage decision: `True` where
the returned row is a direction, `False` where it is a position.
`free_point_decision` is `BundleAdjustment::free_point_decision` as a dict with
the `FreePointDecision` fields as keys (`sigma_px` or `None`,
`observation_count`, `outlier_count`, `decided`, `converged`, `to_finite`,
`to_direction`, `unscored`), and `None` with `free_points_cross=False` or
on a degenerate exit.

A reconstruction read from a `.sfmr` carries its constraints as the
`point_constraints` column, a `uint8` array in the canonical numbering (`0`
free, `1` ranged, `2` held) whatever legend the file stored, and `write_sfmr`
takes it the same way; the file's legend never reaches the dict or
`SfmrReconstruction.point_constraints`.
`sfmtool.fileio.POINT_CONSTRAINT_NAMES` is that canonical numbering as a
tuple of names, so a consumer labels a code with
`POINT_CONSTRAINT_NAMES[code]` rather than hard-coding the numbers, and a caller
building `held=` and `distance=` from the column compares against those codes.

Shapes are validated like `reprojection_residuals`; observation indices out
of range raise. The returned arrays are new (inputs are not mutated from
Python's point of view).

## Testing requirements

- **Perfect-data fixpoint**: synthetic poses/points/observations with zero
  noise stay put (cost already ~0, parameters unchanged to tolerance).
- **Noise recovery**: perturbed poses and points recover the ground truth
  to sub-pixel reprojection on synthetic data; with `opt_f`, a focal
  started 20% off converges to the true value.
- **Robustness**: a contaminated fraction of junk observations does not
  pull the solution (soft-L1 + trim schedule), and the junk ends with
  large `residual_norms` while inliers end small.
- **Trim/track semantics**: an observation set where trimming leaves a
  point with one survivor drops that point's observations from the solve;
  fewer than `min_obs` survivors returns the degenerate all-∞ result with
  the state passed through.
- **Retriangulation re-admission**: a `NaN` point with ≥ 2 observations is
  reborn in round 2 and its observations participate thereafter.
- **World scale**: a scene with noise, perturbed poses, directions and a far
  track, scaled by `×0.01`, `×10` and `×1000`, gives under either policy the
  same representations, storage decision, focal and residuals as at `×1`, and
  poses and points (in camera 0's frame, which takes out the free rotation
  and translation of the world) that are the unscaled ones times the scale.
- **In-front floor**: at every one of those scales, a point a billionth of the
  median depth in front of a camera's centre and a point behind it are
  trimmed, and a point a thousandth of the median depth in front is kept.
  Through the whole adjustment, a two-view point on a camera's centre is left
  out of a one-round solve and comes back unchanged, while the same point a
  thousandth of the median depth out is solved.
- **Pass-through**: images not referenced by any observation are returned
  bit-identical; so are unreferenced points under a single-round schedule
  (multi-round schedules retriangulate them to `NaN` by design).
- **Non-perspective models**: a fisheye scene with perturbed poses
  converges through the central-difference Jacobian fallback under a
  single-round (no-retriangulation) schedule — guarding against a
  zero-Jacobian no-op solve masked by live retriangulation.
- **Focal column exactness off the perspective family**: the analytic
  `(u − cx)/f` column agrees with a central difference of the projection
  in the focal over an `EQUIDISTANT_FISHEYE` field sampled out to
  `θ = 170°` at several azimuths and ray scales, to rounding (no
  truncation allowance) — the derivative of an exactly-linear dependence.
- **Focal release and the fixed-focal gauge, equidistant**: a released
  focal recovers a planted one from a several-percent start on a scene
  whose periphery is past `θ = 90°`; the same solve with `opt_f = false`
  returns the input focal bit-identically; and a multi-coefficient fisheye
  scene (`RADIAL_FISHEYE`) with `opt_f = true` also returns its focal
  bit-identically (the core's degrade, since that model's `∂u/∂f` is not
  `(u − cx)/f`).
- **The `k1` release**: `∂(u, v)/∂k1` matches a central difference of the
  projection in `k1` over a field sampled past `θ = 90°`, to rounding (the
  derivative of an exactly-linear dependence, like the focal column); a
  planted `k1` is recovered from a `k1 = 0` start with the focal fixed and
  again with the focal co-released from a several-percent error; on a scene
  that really is equidistant the released `k1` stays at zero and the
  reconstruction is the fixed-`k1` one (the fixed point the
  EQUIDISTANT_FISHEYE → SIMPLE_RADIAL_FISHEYE(`k1 = 0`) promotion rests on);
  and every other model returns `k1` and the focal unmoved under
  `opt_k1 = true` (the core's degrade).
- **The `k1` step guard**: the admissibility predicate accepts every
  `k1 ≥ 0` and every fold past `θ = π`, rejects a fold inside the imaged
  field, accepts the same `k1` for a camera whose field stops short of it,
  and rejects non-finite steps; end to end, a released solve never returns a
  folded map.
- **The spline release**, on each of the two models that carry a spline: the
  spline columns match a central difference of the projection over the
  model's field and across its domain end (normalized by the sample's
  largest column — an out-of-support column is exactly zero and would
  otherwise measure the finite-difference noise floor); a planted spline is
  recovered from a zero start, coefficient-wise and as a composite map, with
  the focal fixed and again with the focal co-released from a
  several-percent error; on a scene that really is the base model the
  released spline stays at zero and the reconstruction is the fixed-spline
  one (the fixed point the `EQUIDISTANT_FISHEYE` → `SFMTOOL_FISHEYE`(zero
  spline) and `SIMPLE_PINHOLE` → `SFMTOOL_PINHOLE`(zero spline) promotions
  rest on); every other model — and a spline model with an undefined spline
  — returns its parameters unmoved under `opt_bspline = true`, as do the
  spline models under `opt_k1 = true` (the core's degrades); the focal
  release is exercised on its own for each spline model, leaving the
  unreleased coefficients bit for bit; through a field-limited scene on each
  model, the coefficients past the field leave their nonzero starting
  sentinels and continue the supported ones in a straight line, to `10⁻⁶`;
  through a scene with an annulus of radii left out, the one coefficient it
  leaves unsupported is the cubic through the two coefficients on each side,
  to `10⁻⁶`; a spline moves only in a linearization where at least two of
  its coefficients are supported; and directions carry the release where a
  near-axis finite cloud cannot.
- **The spline penalty**: on a nonzero planted spline, the penalty's
  contribution to the gradient is half the central difference of its cost in
  each coefficient and its contribution to the camera block half the second
  difference; every interior row is present, so each end coefficient takes
  exactly one row's curvature; a zero spline has zero penalty and zero
  penalty gradient.
- **The spline step guard**: the admissibility predicate rejects a folded
  spline and non-finite coefficients on either radial coordinate; end to
  end, a released solve never returns a spline violating `1 + δ'(d) > 0` on
  `[0, d_max]`, and where the straight-line continuation of the supported
  coefficients past a field-limited scene would fold, the solve returns a
  monotone spline short of that continuation.
- **The held-distortion retry**: on a scene whose rim observations are folded
  back, so a distortion fitted to them folds, a single iteration with the
  focal and the distortion released returns the spline (and, on a
  `SIMPLE_RADIAL_FISHEYE`, the `k1`) bit for bit while the focal and the poses
  moved; with two cameras of which only one images the folded rim, only that
  camera's spline is held and the other's moves; and the same round run to
  its end reports `converged: false`, says on `progress` that it settled with
  camera 0's spline held, and returns a monotone spline.
- **A weakly supported spline tail**: the tail fixture is a pinhole scene
  with no distortion whose observations thin out toward the domain end, the
  last four knot spans holding at most 1, 3, 12 and 30 of them from the end,
  so that the last of 20 coefficients is reached by one observation or none,
  with ±0.3 px of noise and a starting spline whose last three coefficients
  alternate and give an end slope `1 + δ'(d_max)` of 5.6; the poses and points
  are settled under the default schedule with the lens held.
  - *Support gained within a round.* On a draw where no observation reaches
    the last span at the start and the first step of a round with the focal
    and the spline released carries one into it, two iterations of that
    round move the last coefficient from its start and keep it within 0.25.
  - *The release.* Over four draws at two scene sizes, the focal and the
    spline released under the default schedule return a monotone spline with
    every coefficient within 0.25 (the spline's share of a focal up to 15%
    off is about 0.13), a median residual no more than 0.005 px above a held
    adjustment's from the same start, at a noise level near 0.16 px, and a
    final round that does not report convergence with the end slope under
    0.05, and an end slope under 4 (the start is 5.6). The draws include two
    on which the last coefficient ran away when it was released on gaining
    support within a round with no penalty row, and two on which, without
    the regularization and the held retry, the release ended at an end slope
    of 0.000 and reported convergence there. The end slope itself is not
    bounded away from 0 on this family; the measurements say why
    ([`bundle-adjustment-measurements.md`](bundle-adjustment-measurements.md#a-weakly-supported-spline-tail)).
- **The principal-point release**: on every model, the `cx` and `cy` columns
  match a central difference of the projection in the principal point to
  `10⁻⁶`, and point at the camera's `cx` and `cy` slots; on a distortion-free
  `SIMPLE_PINHOLE` scene whose points fill a 60° field, a principal point
  planted 2% of the width off centre is recovered to within 0.05 px from the
  centre with the focal held and noise-free observations; the same scene
  narrowed to a 5° field converges with the principal point on the image and
  a median residual no higher than with it held; the guard accepts the
  image's edges and rejects a non-finite or off-image principal point, and a
  solve pulled toward a principal point 40 px left of the image ends with it
  still on the image; in a solve with every flag on, a camera whose release
  has only the principal point recovers its own and keeps its focal, and a
  held camera beside it comes back bit for bit.
- **Held image adjustments**: with each image read through its own focal scale
  and principal-point offset, the focal, `k1`, spline and principal-point
  columns of the `SIMPLE_RADIAL_FISHEYE`, `SFMTOOL_PINHOLE` and
  `SFMTOOL_FISHEYE` blocks match a central difference of the adjusted
  projection in the shared parameter; on the 60° principal-point scene with
  observations made through per-image adjustments of up to 1.2% in focal and
  0.8% of the short side in principal point, the solve holding them recovers
  the planted principal point to 0.05 px with every residual and the noise
  level under `10⁻³` px, while the same pixels read without the adjustments
  leave residuals over 0.2 px; on the planted pinhole spline's scene with
  per-image adjustments the spline is recovered from none; and all-zero
  adjustments give the solve with none bit for bit.
- **The per-image release**
  ([`bundle_adjust/tests/per_image.rs`](../../../crates/sfmtool-core/src/geometry/bundle_adjust/tests/per_image.rs)),
  on scenes whose images look from a cap of directions, each at its own point
  and rolled, since images whose axes meet in one point leave the per-image
  focal a family of exact solutions:
  - the log focal scale's and the offset's columns match a central difference
    of the adjusted projection, in `t = ln(1 + s)` for the focal, on every
    model each is released on, and point at the image's own slots;
  - from noise-free observations made through per-image focal scales and
    offsets about a common part that is not zero (1% long, `(6, −4)` px), every
    image's intrinsics are recovered to `10⁻⁷`, the camera holding the planted
    mean, the geometric mean of `1 + s` for the focal, and the adjustments
    averaging zero to rounding, the focal's in `t`;
  - the derived width recovers a planted spread of offsets (0.4% of the short
    side over forty images with half a pixel of noise) within `[0.75, 1.35]` of
    it, and, where each image's variance is comparable to the spread's square
    (25 observations at 0.7 px), within `[0.7, 1.4]`, the width after five more
    rounds on the same observations within 5% of the first; the derived focal
    width recovers a planted spread of 1% in `t` over forty images within
    `[0.75, 1.35]` of it;
  - one more image of eight observations through an offset twelve spreads out
    moves the width by less than 15%;
  - images that do not vary derive the floor; `fixed` widths are used as given
    and nothing is derived; a one-round schedule derives once;
  - images that share one offset, `(6, −4)` px from the starting camera's principal
    point, solved in one round under a fixed width far narrower than it, move
    together, and the camera ends within 1.5 px of their principal point: the
    prior holds the images to their mean, not to zero; likewise images that
    share one focal 3% longer than the starting camera's end with the camera
    within 0.2% of it;
  - an image with no observation keeps its intrinsics through the re-centring
    while its camera moves;
  - the shared and per-image release of one kind gives the per-image release's
    solve bit for bit, and flags on with nothing released per image give the
    solve with the flags off bit for bit;
  - the focal guard refuses a log focal scale that is not finite or leaves the
    focal not finite and above `1e-6`, and an image whose observations pull its
    principal point off the image ends on it;
  - the derivation alone, on samples made as a round makes them, undoes a pull
    to under 60% of the spread and recovers the spread within 15%, and reads
    the focal in `t`: the same deviations given as log focal scales and as
    principal-point offsets derive the same spread.
- **Directions carry the `k1` release**: on a scene whose finite cloud sits near the
  optical axis (no `θ³` signal) and whose far field is marked at infinity,
  `opt_k1` recovers the planted curvature — and the same solve with the
  direction observations removed does not.
- **Memory order**: Fortran-ordered inputs to the binding produce the same
  result as C-ordered ones (guards the `to_contiguous!` zero-copy path
  against silent transposition).
- **Binding behavior**: the Python binding reproduces the kernel's
  behavior on analogous synthetic scenes (`tests/rust_bindings/`).
- **Several cameras**:
  - *Parity.* Every single-camera test runs through `BaCameras::shared`, and a
    scene run through an explicit one-entry `BaCameras` matches it to the bit.
  - *Two models.* A scene with a `PINHOLE` and an `OPENCV_FISHEYE` camera,
    images from each, and perturbed poses and points converges to sub-pixel
    reprojection; the same scene with every image read through the pinhole
    does not, which shows each observation is read through its own camera.
  - *Per-camera focal release.* Two releasable cameras with different planted
    focal lengths, both started several percent off, recover each its own focal
    under `opt_f`; with one releasable and one not, the releasable one moves and
    the other comes back bit for bit.
  - *Independent blocks.* Cameras no image uses come back bit for bit under
    every release, whatever their models admit, while the used camera is solved.
  - *Two distortion releases in one solve.* A `SIMPLE_RADIAL_FISHEYE` releasing
    `k1` and a `SFMTOOL_FISHEYE` releasing its spline recover a planted `k1` and
    a planted spline together.
  - *Directions past 90°.* On an equidistant fisheye scene wider than 180°,
    directions observed at `θ = 100°` survive the trim and recover the rotation
    of an image whose only observations they are; on a perspective camera a
    direction behind the image plane is trimmed.
  - *The storage decision reads each lens.* A decision over images split
    between two focal lengths measures the level the residuals of both cameras
    give, and decides each track as `bearing_score` does on rays weighted
    through the camera of each sighting's image.
  - *Shape checks.* An index past the camera list, an `image_camera` of the
    wrong length and an empty list panic; the binding raises `ValueError` for
    each, and names the camera a refused release does not admit.

## Points at infinity

There is one staged loop, not two: direction handling is a per-point branch
inside it, so with nothing marked the loop *is* the finite-only solve.

A point at infinity is a pure direction: its observations depend on the
observing image's rotation and its camera's model, never on any
translation. Supplying far-field tracks as directions therefore pins
rotations (and, under `opt_f`, the focal) without touching the
depth/translation side of the solve — exactly the coupling that lets a
near-planar or low-parallax scene trade rotation bends against a wrong
focal.

### State and inputs

- A per-point mask `point_at_infinity: &[bool]` (`n_pt`) marks direction
  points. A marked point's `X_p` slot holds a **world-frame direction**;
  the kernel normalizes it on input and returns it normalized. `NaN`
  directions are allowed and behave like `NaN` finite points (invalid
  until re-estimated). An absent mask (binding: `point_at_infinity=None`)
  is normalized to an all-`false` mask at the entry point, and an all-`false`
  mask skips every direction branch — so both are exactly the finite-only
  solve.
- Directions live in the same `points` array; the mask is the only
  distinction. The caller's array is not modified: the representation each
  point ended with comes back as the result's own `point_at_infinity`, which
  is the input mask unless a free point crossed (see "Point constraints").

### Residuals and derivatives

A direction projects like a point at infinite depth: `uv_pred =
ray_to_pixel(R_i · d)`. The residual is the same pixel difference as a
finite observation — same units, same soft-L1 loss, same trim thresholds.
Whether a direction is "in front" is model-aware, like the finite in-front
measure: for the perspective family it is `(R_i · d)_z < 0` (canonical −Z
forward), and for a ray-path model, whose domain runs past `θ = 90°`, it is
that `ray_to_pixel(R_i · d)` is defined, so a direction a wide fisheye sees past
90° off-axis is in front. The trim reads the same measure it reads for a finite
point, the range `‖R_i · d‖ = 1` for a ray-path model, against a floor of zero,
and the domain test is the residual. A behind-camera or out-of-domain direction
contributes the standard `(1e6, 0)` penalized residual with a zero Jacobian row.

- **Parameters.** A direction perturbs in the 2-DOF tangent plane of the
  unit sphere: `d ← normalize(d + B(d) · δ)` with `B(d)` an orthonormal
  basis of `d⊥` rebuilt at each linearization. Its Schur block is 2×2
  where a finite point's is 3×3; the translation Jacobian block is zero;
  the rotation block is `−[R·d]ₓ` composed with the same projection
  Jacobian as finite points, and the `opt_f` derivative applies
  unchanged.
- **Translation observability.** Infinity observations constrain no
  translation, so an image's translation is frozen for a round when no
  surviving observation of it carries a translation Jacobian: every
  direction, free or held, carries none, while a finite point does whoever
  owns it (a held finite point and a ranged point at a finite distance
  included). Its rotation still updates; otherwise the reduced camera system
  would carry a zero-curvature translation block. The `min_obs` degenerate-exit floor is independent of
  that and counts **every trim survivor**, finite and direction alike: it
  measures whether the round retained enough evidence to solve on, and a
  direction constrains the rotations and its camera's lens parameters just
  as a finite observation does. A directions-only observation set therefore
  runs at the default floor.

### Staged-loop semantics

- **Trim** treats direction observations exactly like finite ones (pixel
  threshold, `min_track` survivors per point); the in-front check is the
  model-aware test above, against a floor of zero, the in-front floor's
  value at zero inverse depth.
- **Re-estimation (rounds after the first).** Where finite points
  retriangulate, a direction re-estimates in closed form as the
  normalized mean of its observations' back-rotated rays
  `R_iᵀ · pixel_to_ray(uv)` at the current rotations. A direction track
  with fewer than 2 observations becomes `NaN`, mirroring finite tracks. Both
  families are one call of the retriangulation operation with the
  adjustment's settings (marks on, few absent, every other rule off), see
  [triangulation-rules.md](../reconstruction/triangulation-rules.md).

### Binding

`bundle_adjust(..., point_at_infinity=None)` — optional `(n_pt,)` bool
array. The returned `points` rows of marked points are unit directions.
All other shapes, validation, and outputs are unchanged.

### Testing requirements (additional)

- **Regression**: an absent mask and an all-`false` mask agree bit for bit
  on the existing synthetic scenes (the entry point's normalization), and
  appending a marked direction row that no observation references leaves
  every finite result bit-identical (the direction branches stay inert when
  their flag is unset — the guard on the single-loop reduction).
- **Direction fixpoint and recovery**: noiseless direction observations
  stay put; perturbed rotations recover ground truth against a far-field
  direction set to sub-pixel reprojection.
- **Rotation lock under `opt_f`**: on a synthetic low-parallax scene
  (near-planar finite cloud) where a focal started well off converges
  wrongly without directions, adding far-field direction tracks recovers
  the true focal.
- **Frozen translation**: an image observing only directions returns its
  translation bit-identical while its rotation refines.
- **Re-estimation**: a `NaN` direction with ≥ 2 observations is reborn in
  round 2 as the mean back-rotated ray.
- **Memory order and binding parity** as for the kernel above.

## Point constraints

`point_at_infinity` says what a point's coordinate *is*; `constraints` says who
*owns* it. The two are orthogonal, and every point carries one of three
constraints.

- **Free.** The solve owns the point. Under `FreePointPolicy::cross` it is
  solved in inverse depth and its representation is whatever its rays support
  at the geometry the solve ends at, decided once at the end; with the policy
  off it keeps the representation the caller handed in.
- **Ranged.** The caller owns one number, the point's distance `r` from a
  reference it names; the solve owns the direction. The point is `X = O + r · d`
  with `d` a unit vector, and `d` is its only parameter -- two degrees of
  freedom on the sphere. `r = ∞` needs no reference and is exactly a direction.
- **Held.** The caller owns the point. Its coordinate, finite or direction, is
  fixed for the whole solve. Its observations still form residuals and still
  feed the camera and lens blocks, but the point has no parameters and no Schur
  block, and the re-estimation skips it.

The constraints nest: a held point is a ranged point that has also given up its
direction, and a ranged point at an infinite distance is what the mask alone
calls a marked point. `protected` stays orthogonal to both, being about whether
an observation can be trimmed, not about whether a point can move. Trim,
`min_track` and `min_obs` treat a ranged or held point's observations exactly
like any other's.

A caller states only the points it owns, and leaves the rest free:

```rust
let mut cons = PointConstraints::all_free(points.len());
cons.hold(survey_marker);                        // known in the solve's frame
cons.constrain_distance(spire, 1045.0, Some(DistanceReference::Image(shot)));
cons.constrain_distance(sky, f64::INFINITY, None);  // a bearing, no reference
bundle_adjust(&BaCameras::shared(&cam, quats.len()),
              quats, trans, points, uv, obs_img, obs_pt,
              None, Some(&cons),
              FreePointPolicy { cross: true },
              None, DEFAULT_PROTECTED_LOSS_SCALE,
              true, false, false, false, false, false, None,
              &DEFAULT_SCHEDULE, 60, 2, 12, &Progress::none());
```

An absent `constraints` is every point free, and with `FreePointPolicy::NO_CROSS`
(`cross = false`) every free point keeps the representation it is handed in,
as the sections above describe. The Rust interface is
[bundle_adjust.rs](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs)
(`PointConstraint`, `PointConstraints`, `DistanceReference`, `FreePointPolicy`,
`FreePointDecision`);
the Python binding takes the same three constraints as flat per-point arrays
(see [Bindings](#bindings)), and a reconstruction carries them in the `.sfmr`
constraint triple (see
[Per-point constraints](../../formats/sfmr-file-format.md#per-point-constraints-optional-version-7)).

### Free points: inverse depth and the storage decision

```rust
pub struct FreePointPolicy {
    pub cross: bool, // solve free points in inverse depth and decide them
}

impl FreePointPolicy {
    pub const CROSS: FreePointPolicy;    // the default
    pub const NO_CROSS: FreePointPolicy; // the opt-out: the caller's representation stands
}

pub struct FreePointDecision {
    pub sigma_px: Option<f64>,   // the level measured at the end; None: no finite observation
    pub observation_count: usize,
    pub outlier_count: usize,
    pub decided: bool,           // the test was read (a level, and not cancelled)
    pub converged: bool,         // the final round met its convergence test
    pub to_finite: usize,        // free points handed in as directions, stored finite
    pub to_direction: usize,     // free points handed in as positions, stored as directions
    pub unscored: usize,         // free points with too few kept observations to score
}
```

Under `cross` every free point is solved in **inverse depth**, so that within a
round it can move between a near position and infinity, and whether it is
stored as a position or as a direction is decided once, at the end of the solve,
by the point-or-bearing test
([batch-triangulation-api.md](../reconstruction/batch-triangulation-api.md)
§ "Point or bearing") at the noise level the final round's residuals measure.
`BundleAdjustment::free_point_decision` reports that decision; it is `None`
with the crossing off, and when the solve exits degenerate (see "The staged
loop"), since no final round was solved to measure a noise level on. A free
point that ends as a direction comes back as a unit row, as any direction
does. Ranged and held points keep their own parametrisations and are not
decided.

The crossing is the default, `FreePointPolicy::default()` being
`FreePointPolicy::CROSS`, so every caller in the crate crosses unless it states
otherwise: `rotation_init`, `grow_reconstruction` (whose `GrowOptions::
free_points` carries the switch), the reconstruction-level adjustment (whose
`BundleAdjustOptions::free_points` carries it) and through that `sfm xform
--bundle-adjust` and the viewer's Bundle Adjust. Each of those but
`rotation_init` lets its own caller opt out with `FreePointPolicy::NO_CROSS`:
`--bundle-adjust cross=off`, the Bundle Adjust dialog's crossing checkbox, the
`bundle_adjust` wire tool's `free_points_cross`, and the bindings'
`free_points_cross=False`. The default is the module's rather than each
caller's because the decision is the representation every consumer of a
reconstruction should read: a caller that holds the representation it was
handed is the one making a choice, and it says so.

The kernel says what the decision did through its `progress`, after the
rounds: `free points decided at noise 0.412 px: 3 to finite, 25 to directions`,
with `, 2 not scored: too few kept observations` appended when some free
points could not be scored, `; the final round stopped on its iteration
budget` when the final round ran out of iterations, and `; the final round
settled with a distortion held at its step guard` when it settled at the guard
(see "Termination"), or `free points not decided: the final round kept no
observation of a finite point`. An empty schedule, which only measures,
says nothing. `sfm xform --bundle-adjust` prints its own line from the returned
decision instead (`Free points decided at 0.412 px: …`, without `noise`).

#### The parametrisation

A free point is an anchor `a`, a unit direction `u` and an inverse depth
`ρ ≥ 0`, with the point at `a + u/ρ`. Its observation from image `i` projects
the ray

```
p̃ = R_i·(u + ρ·a) + ρ·t_i
```

which is `ρ·(R_i·X + t_i)`, a positive multiple of the camera-frame point, so it
projects to the same pixel and lies on the same side of the camera. At `ρ = 0`
it is `R_i·u`, the direction's own ray. Every projection, Jacobian and lens
column the solve reads is homogeneous of degree zero in the ray, so the solve
reads `p̃` directly and never forms the point. This is the parametrisation of
the test's own point fit (`fit_point_and_bearing`).

- **Parameters.** `u` perturbs in its 2-DOF tangent plane, `u ← normalize(u +
  B(u)·δ)`, as a direction does, and `ρ ← ρ + δρ`: three slots, the same as a
  Euclidean point's, so the point block stays 3×3 and the Schur complement is
  unchanged in shape.
- **Jacobian.** With `J` the projection block at `p̃`, the rotation block is
  `−J·[R_i·(u + ρ·a)]ₓ`, the translation block `ρ·J`, the point block
  `[J·R_i·b₁, J·R_i·b₂, J·(R_i·a + t_i)]`, the last column being the anchor as
  image `i` sees it, and the lens columns are read at `p̃` unchanged. All are
  analytic and are checked against a central difference, at `ρ = 0` included.
- **The bound.** `ρ` is clamped at zero inside the damping ladder, as the point
  fit clamps it. A step that would take a positive `ρ` below zero stops it at
  zero. A point already at zero whose gradient asks for a negative `ρ` has its
  `ρ` column dropped for that iteration, before the reduced system is formed,
  so it is eliminated over its two tangent slots as a direction is and the
  camera step is not solved as though it could pass through infinity. A point at
  zero whose joint step still goes below zero takes the step in `u` alone. An
  accepted state never holds a negative `ρ` (asserted in debug builds). Without
  the bound, rays that diverge would be fitted by a negative `ρ`, a point
  behind the cameras.
- **The anchor.** The centroid of the centres of the images observing the
  point, one term per observation, at the poses the round starts from, and
  fixed for the round. A point that sits exactly on its anchor has no
  direction from it and is solved as a position for that round.
- **In and out of the round.** Between rounds the arrays hold the ordinary
  representation, which is what the trim, the re-estimation and the caller
  read: a position `X` enters as `u = (X − a)/‖X − a‖`, `ρ = 1/‖X − a‖`, a
  direction `d` as `u = d`, `ρ = 0`, and the round hands back a position where
  `ρ > 0` and the direction `u` where `ρ = 0`. Soft-L1, the trim, `min_track`
  and protection apply exactly as to any other point.
- **Translation observability.** A point at `ρ = 0` carries no translation
  column, as a direction does not, so an image is pinned as before when none of
  its kept observations carries one, re-read every iteration rather than once a
  round since `ρ` moves.
- **Re-estimation between rounds.** A free point starts the next round from
  whichever of three states fits its whole track best: the state the last
  round left it in, its midpoint over all its observations, and its mean ray
  at `ρ = 0`. The fit is the sum over the track's observations of the squared
  residual capped at the next round's `trim_px`, and a tie keeps the earlier
  state in that order. A state in the other representation than the last
  round left is taken only when at least `min_track` of the track's
  observations reproject under that trim from it, the number the next round's
  trim needs to keep the track; otherwise the last state stands. A point with
  no estimate (`NaN`) takes its midpoint, or its mean ray where that costs
  less, with no guard, since it has no representation of its own to keep. The
  midpoint is read with
  `cheirality` on, so one that lands behind an observing camera is the mean ray
  ([triangulation-rules.md](../reconstruction/triangulation-rules.md)); a track
  with fewer than two usable observations has no estimate and is `NaN`, as any
  track is. That is a starting value, not a decision: the next round's solve
  moves `ρ` from there. Neither closed form is right for every track. The
  midpoint of nearly parallel rays can land near the cameras, where its
  residuals put the whole track past the trim, so a far point the solve has
  just fitted is kept as fitted; the mean ray of a near point is off by its
  parallax, so a point handed in as a direction whose rays fan out, whose
  observations the first round trimmed, starts the next round at its midpoint
  and is solved there. The capped sum reads the track the way the next trim
  will, so a single gross outlier costs no more than any other observation the
  trim will drop. The guard on a change of representation is for a rough
  start, where every state leaves most of a track past the trim and the mean
  ray can cost least by bringing a single observation under it: the next trim
  then starves the track, and the decision, which cannot score it, puts it
  back to the row it was handed in with, discarding what the rounds made of
  it. With the guard such a point comes back as the solve left it. On the
  seoul bull `sift_files` solve from 1° and 5% at five iterations a round,
  without the guard 62 positions become directions this way at the second
  round and 60 of them come back unscored and are reset to their input rows;
  with it 3 become directions and none is reset. The result holds 9 directions
  either way, with a median residual of 2.814 px and 1,660 observations under
  4 px against 2.819 px and 1,656 without the guard; the Kerry Park solve and
  the Kerry Park ground truth from the same start store the same 225 and 27 directions either way.

#### The storage decision

After the last round, the level `σ` is measured with the estimator of the stored
measure: the RMS per-axis residual over the final round's kept observations
of finite points at the state the solve ended at, each camera's residuals gated
at `OUTLIER_GATE` robust spreads, no degrees-of-freedom correction, never under
the cameras' keypoint resolution
([batch-triangulation-api.md](../reconstruction/batch-triangulation-api.md)
§ "The measured noise level"). A direction's residuals are left out. Each free
track with an estimate is then scored over the same set: the observations of it
the final round kept, with rays and noise weights from `observed_ray` through
the camera of each observing image as the solve ended, and `is_finite` at
`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD` decides:

- **A finite verdict** keeps the position the solve placed where `ρ > 0`. Where
  the solve left the point at `ρ = 0`, it is placed at the test's own point fit
  (`fit_point_and_bearing` with the default soft-L1), where that point lies in
  front of every observing camera; otherwise the direction stands.
- **A bearing verdict** stores the bearing the test fits, where that bearing
  lies in front of every observing camera, whether the solve left the point at
  a position or at `ρ = 0`: the stored bearing does not depend on which side of
  the bound the solve's own fit ended. A bearing behind a camera describes no
  sighting there, and what the solve left stands.
- **Too few kept observations to score.** A free point whose kept
  observations give fewer than two usable rays -- its track trimmed below
  `min_track`, or down to one observation that protection or a `min_track` of
  one kept -- is not scored. It is stored in the representation the caller
  handed it in: as the solve left it where that is the same, and as handed in
  where the rounds moved it to the other, so that a point the test never read
  is never stored in a representation nothing chose. A point handed in with no
  estimate (`NaN`) has no row to go back to and keeps the representation the
  re-estimation gave it. It is counted in `unscored` and in neither
  `to_finite` nor `to_direction`.

The decision reads the observations the solve fitted and nothing else. An
observation the trim, the in-front floor or `min_track` rejected changes neither
a verdict nor a stored bearing, so a trimmed outlier does not pull a bearing off
or turn it finite, and a track whose observations the trim dropped as junk is
not stored as the bearing that best fits that junk. The level and the scores are
then read from one set of observations at one state.

`residual_norms` are read after the decision, at the representation stored,
over every supplied observation as always; the trim and its counts are the
solve's and the decision does not change them. A solve with no kept observation
of a finite point at the end has no level and decides nothing (`sigma_px:
None`); nor does a cancelled one; each free point is then stored as the solve
left it, a position where `ρ > 0` and a direction where `ρ = 0`, and `unscored`
is zero. `to_finite` and `to_direction` count the free
points stored in the other representation than the caller handed in.

The decision changes representations without refitting the poses. A point the
solve left at `ρ > 0` and the decision stores as a bearing leaves poses that
were fitted to it as a finite point: re-solving the result of the clean
[Kerry Park ground truth](../../../test-data/images/kerry_park/kerry_park_ground_truth_v01.sfmr)
with the crossing off, every point held in the representation the decision
stored, lowers its cost by 17.3 noise units and takes the scores of the six
far points the decision made finite
([measurements](bundle-adjustment-measurements.md#the-default-schedule-against-the-crossing-between-rounds))
back under the
threshold of 25, to 18.8 to 24.2. No refit follows the decision, because a solve
that carries a point in the representation just chosen is the direction bias
described below ("Why inverse depth"), which pulls each score toward the side
it was put on; the poses the inverse-depth solve leaves are the ones fitted with
no representation imposed.

#### Why inverse depth

The alternative, called the crossing between rounds, carries each free point
as a position or a direction for a whole round and re-decides it on the test
between rounds. It is rejected because the geometry of a round is then solved
with the point in the representation it has, and the solve fits that
representation: a borderline track's score moves toward the representation it
starts in, so the track keeps it, and one round bends the poses to fit a far
track as whichever it is given. A marginal track needs several rounds to
settle. A round that stops on its iteration budget far from convergence
measures a level that is mostly pose error, makes points with a depth
directions at it, and the next round bends the poses to fit them
([measurements](bundle-adjustment-measurements.md#the-crossing-between-rounds)).

In inverse depth `ρ = 0` is a value the solve can reach, so a point carries no
representation through a round and the poses are not fitted to one chosen
before it; a far point stays well conditioned because its inverse depth is
close to linear in the observations; and nothing is decided at a round
boundary, so a rough round cannot demote anything. On synthetic far tracks
spanning the threshold, the verdict at the result agrees with the verdict at
the true poses on more tracks solved in inverse depth than carried as positions
or as directions. On the default schedule over the four inputs of the
point-or-bearing work, every converged solve stores every point as the test at
the level the decision read says; read at the result's own level, the Kerry
Park solve disagrees on 2 to 10 points, where the crossing between rounds left
27 to 34
([measurements](bundle-adjustment-measurements.md#the-default-schedule-against-the-crossing-between-rounds)).

#### Decisions

**Translation observability needs only the pinning directions already had.** For
`ρ > 0` the translation column is the Euclidean point's (`ρ·J` at `p̃ = ρ·p_cam`
is `J` at `p_cam`), and the Schur complement is invariant under an invertible
reparametrisation of a point, so the reduced camera system is the Euclidean one
up to the Marquardt damping; on the images observing the most far points the
translation blocks of the two agree to the printed digits
([measurements](bundle-adjustment-measurements.md#translation-observability-in-inverse-depth)).
Only `ρ = 0` exactly makes the column zero, and that is the case the existing
pin covers; no damping that reads the column's size is needed.

**The anchor is re-read at the start of every round.** The anchor only chooses
the parametrisation, but the bound at `ρ = 0` and the singularity at the anchor
are places a point cannot move through, and an anchor left at the input poses
can sit away from cameras a rough start has moved. Because the re-estimation
reads a position between rounds anyway, `ρ` keeping its meaning across rounds
buys nothing. From starts up to 1° and 5% the two choices store the same
representation for every point and agree on the median residual to 0.001 px;
from 4° and 20% the anchor left at the input poses stops the second round early
and keeps fewer observations under 4 px
([measurements](bundle-adjustment-measurements.md#re-reading-the-anchor-every-round)).

**The decision is read at the end of an unconverged solve too.** A final round
that stops on its iteration budget measures a level that still carries pose
error, which errs toward a direction, and `converged: false` says so. Not
deciding, storing each point as the solve left it, is further from the
converged answer at every budget measured: noise puts the best `ρ` of a far
point above zero about half the time, so the solve's own `ρ` leaves nearly
every far point finite. A rule that decided only after convergence would also
leave undecided a run whose level has already settled at the converged one
([measurements](bundle-adjustment-measurements.md#deciding-at-the-end-of-an-unconverged-solve)).

**At the default budget an unconverged end is rare, and its level is the
converged one.** Every caller runs 60 iterations a round. Of 48 solves of
thirteen inputs through the reconstruction-level adjustment, with the lenses
released, 6 end with an unconverged final round, none of them from a clean
start, and in each the level is within 2% of the level the same input reaches
when every round converges
([measurements](bundle-adjustment-measurements.md#unconverged-ends-at-the-default-budget)).

**At short budgets the remaining error is the unconverged state.** At 5 to 20
iterations a round, deciding at the converged level instead of the measured one
removes a small part of the disagreement with the converged solve; the rest
comes from poses and points that are not yet the converged ones, which no level
corrects. Deciding at the previous round's level or at a median-based level, or
never demoting a point handed in as a position, disagrees more. Running the
last round once more disagrees a little less in total but more on some inputs,
and far more on a solve that has collapsed; what it approximates is a larger
`max_iters`, which is the caller's to choose
([measurements](bundle-adjustment-measurements.md#remedies-at-short-budgets)).
So the decision is read at the final round's measured level whether or not that
round converged, `converged: false` and the progress line saying when it did
not.

**`converged` reports how the final round ended, not success.** It says whether
the final round met its convergence test, rather than stopping on its iteration
budget or settling with a distortion held at its step guard, and a solve that
has collapsed meets it as readily as one that has recovered, so its points are
decided all the same. A collapse shows in the final round's kept count and the
median residual, not in
`converged`, and it is not specific to the crossing: with the crossing off the
same rough starts collapse too
([measurements](bundle-adjustment-measurements.md#collapsed-solves-that-report-convergence)).
`FreePointDecision::observation_count` is the number of observations the level
was measured over, which a caller can set against the number it supplied.

**Rough starts with the lenses released.** The crossing changes which rough
starts the reconstruction-level adjustment recovers from, in both directions,
and with every lens released neither setting recovers reliably from 4° and 20%
([measurements](bundle-adjustment-measurements.md#rough-starts-with-the-lenses-released)).

**Cost is the solve's own.** The crossing changes how many iterations the
rounds take, not what an iteration costs: an iteration costs the same in either
parametrisation, and the re-estimation and the decision under the crossing add
hundredths of a second a round on the dino global solve. Over the callers
and inputs measured the solve with the crossing takes 0.43 to 2.68 times the
time it takes with the crossing off, and the slower runs are the ones whose
rounds take more iterations
([measurements](bundle-adjustment-measurements.md#cost)).

**Inverse depth is the crossing, not a second switch.** The two halves are not
useful apart. Inverse depth without the decision stores the solve's own `ρ`,
which leaves almost every far point finite (see the decision at an unconverged
end, above), and a decision on a point carried through the solve as a position
or a direction reads the score that representation has pulled toward itself
("Why inverse depth"). With the switch off every free point keeps the
representation it is handed in, which is what a caller that opts out relies
on.

**The level is measured, not the loss scale.** `loss_scale` is a schedule
constant, chosen before anything is measured, and one schedule runs over
captures whose noise differs fivefold: the levels the decision reads on the
clean inputs run from 0.19 px on the Kerry Park solve to 1.05 px on
`dino_dog_toy`. The level comes from the final round's kept observations, the
residuals the solve has just minimised, whose trim capped every residual it
kept at 4 px by default, which bounds what a gross outlier the gate passes can
add. The level `sfm analyze --depth-reliability` reports for a result is the
stored measure over every observation, including those the adjustment's last
trim left out, so it can be higher than the level the decision read, and a
point it lists as disagreeing with its stored representation can be one the
decision stored as the test said
([measurements](bundle-adjustment-measurements.md#disagreements-left-at-a-converged-end)).

### Ranged points: a direction at a distance

A reference is a function of the camera poses, never a fixed world coordinate:
the adjustment's gauge is free, so a distance measured from a point the cameras
can move away from constrains nothing about the cameras. The two forms are one
image's camera centre, `O = C_k = −R_kᵀ · t_k`, and the mean of a set of them,
`O = (1/|K|) · Σ_{k∈K} C_k`, a capture station whose frames sit close together
and none of which is the survey point on its own.

Residual and derivatives at a finite `r`:

- `uv = ray_to_pixel(R_i · X + t_i)` with `X = O + r · d`, the finite
  projection with the point's position substituted. The caller's row of
  `points` is a position, read as `d = normalize(X − O)` at the round's poses,
  so a row that does not sit at exactly `r` from the reference is snapped onto
  the sphere the distance names; the row written back is `O + r · d` at the
  poses the round settled on.
- **Point block.** `d` perturbs in its 2-DOF tangent plane exactly as a
  direction does, and the block is `r · J_X · B(d)` with `J_X = ∂uv/∂X` the
  finite point's position Jacobian. Its Schur block is 2×2.
- **Observing camera's block.** The rotation, translation and lens blocks of a
  finite point at `X`, unchanged.
- **Reference images' blocks.** `X` moves with `O`, so every observation of the
  point also contributes `J_X · ∂C_k/∂(pose_k) / |K|` to image `k`'s camera
  block for every `k ∈ K`, with `∂C_k/∂t_k = −R_kᵀ` and
  `∂C_k/∂ω_k = −R_kᵀ · [t_k]ₓ`, the derivative of `−Rᵀ·t` under this kernel's
  own rotation update `R ← exp(ω)·R`, which leaves `C' = −Rᵀ·exp(−ω)·t`. Where
  the observing camera is itself in `K` the two contributions land in the same
  block and add. A reference image no surviving observation touches has no slot
  in the round's reduced system, so the round holds it fixed and its centre
  enters `O` as a constant.
- **At `r = ∞`** every one of these is the direction case: the point block is
  the tangent Jacobian, the translation block and the reference blocks vanish,
  and the re-estimation returns the normalized mean of the back-rotated rays.

Because `r` is held in the solve's own units, ranged points carry metric scale
into an adjustment that otherwise has none: several ranged points on one
reference set fix the scale gauge, and a distance that disagrees with the
caller's other scale evidence shows up as residual rather than being absorbed.
The converse is worth expecting: a solve carrying one distance moves the whole
reconstruction under it until the scale fits.

Re-estimation between rounds keeps `r` and re-solves `d` through the
retriangulation operation's `distance` rule at the origin the reference
resolves to at the round's poses
([triangulation-rules.md](../reconstruction/triangulation-rules.md)).

### Held points: residuals without parameters

A held point's observations project exactly as any observation of its
representation does, and the camera Jacobian blocks are formed the same way. The
point's own Jacobian block is absent: nothing is accumulated into a point block,
no Schur complement is taken for it, no update is back-substituted, and its
coordinates are copied through to the output untouched. A held direction row is
normalized on input like any direction row; a held finite row passes through as
it came.

Trim applies to a held point's observations as to any other's; a held point does
not make an observation `protected`, and a caller that wants both marks both. A
held point that loses every observation contributes nothing and is still
returned unchanged. An image observing one held finite point and otherwise only
directions keeps its translation live, which is what a surveyed landmark is for.

### Testing requirements (additional)

- **Parity**: on a fixture mixing finite points, directions and protected
  observations, an absent `constraints` and an all-free one under
  `FreePointPolicy::NO_CROSS` agree on every output field to the bit (poses,
  points, focal, residual norms and the reported representation), and neither
  reports a storage decision.
- **The crossing off**: under `FreePointPolicy::NO_CROSS`, on a fixture with a
  released focal, directions, a far track, pixel noise, perturbed poses,
  protected observations, a held point and a ranged one, sums over the points,
  poses, residual norms and focal match values recorded for the fixture, to
  `1e-12` relative (so that a platform's `libm` rounding a transcendental
  differently in its last place does not fail it), and every marked point comes
  back in the representation it went in with. A change to what the kernel
  computes with the crossing off is a change to those recorded values.
- **The ranged Jacobian**: the analytic blocks of every observation, assembled
  into a dense Jacobian, match a central difference of the whole residual
  vector on a small ranged scene, with the reference image apart from the
  observers, among them, and as the mean of two.
- **The inverse-depth Jacobian**: the same check for a free point in inverse
  depth, in every pose slot and in `(δ₁, δ₂, δρ)`, at a finite `ρ` and at
  `ρ = 0`, where the difference reads both sides of the bound and the
  translation columns are exactly zero.
- **The bound**: a track whose rays diverge, so that the best inverse depth is
  negative, comes out of a round's solve a unit direction in front of every
  observing camera, from a finite start and from a direction; the solve asserts
  in debug builds that no accepted `ρ` is negative, which the test trips when
  the clamp is removed.
- **Both directions**: a near cloud started at infinity comes back finite at
  its true positions, and a far track started finite in a capture with a third
  of a pixel of noise comes back as a unit direction, with the reported
  representation matching the row and the decision counting one direction. On
  exact pixels the measured level is the keypoint resolution and any parallax
  is a depth, so the far case needs the noise.
- **The boundary is the measured noise**: over ten cameras and 300 points, the
  same track 3,000 units out is finite with 0.09 px of noise and a bearing with
  0.76 px, from either starting representation, and the measured level follows
  the noise put in.
- **The storage decision, by hand**: one round with no iterations and no trim,
  so the state the decision reads is the input: its level is the gated RMS of
  the finite residuals computed by hand (a 40 px mismatch is the one outlier),
  and every verdict and bearing is that of `bearing_score` and `is_finite` on
  rays `observed_ray` builds through each image's own camera, over two cameras
  of different focal length; the decision reports that it did not converge.
- **No finite observation, no decision**: a directions-only solve reports no
  level, decides nothing and changes no representation.
- **A borderline verdict does not depend on the start**: across distances whose
  score at 0.76 px runs from about 60 to about 8, the track ends with the same
  verdict from either starting representation, at the same bearing to 2e-5 rad
  or the same position to 1e-4 in the first camera's frame (two solves can end
  in gauges a small rotation apart), and the sweep spans both verdicts.
- **A far point from a wrong depth**: a track 20,000 units out in 0.76 px of
  noise, started 100 units out and at infinity, ends a bearing both ways at the
  same bearing; one 1,000 units out in 0.09 px, started ten times too far and
  at infinity, ends the same point both ways, near its true distance.
- **A rough start**: from rotations perturbed by up to 0.6°, with two
  iterations a round the decision is taken and reports no convergence, every
  near point ends finite and nothing is `NaN`; with 60 it converges and the far
  tracks end directions as well.
- **An unconverged round keeps points with a depth**: from rotations perturbed
  by up to about 6° about each axis and translations by up to 0.2 units along
  each, over six rounds, the thirty points 250 to 395 units out, whose rays
  carry a depth at the converged level but not at three pixels, and every near
  point end finite with three iterations a round and with 60, and the decision
  makes no point a direction.
- **Held points**: their coordinates come back to the bit while free points
  move, and their observations still carry residuals; an image whose only
  finite evidence is one held point solves its translation, where the same
  landmark as a direction leaves the translation frozen to the bit.
- **Ranged points**: an infinite distance reproduces a marked direction held
  as one (`FreePointPolicy::NO_CROSS`) bit for bit; a finite one comes back at
  exactly its distance from the reference read at the final pose; and a
  landmark started at a wrong bearing but its true distance recovers the
  bearing the reference image sees, where the same track free converges to a
  wrong depth.
- **The binding**: with `free_points_cross=False`, its off position (absent
  arguments, and explicitly-off ones) agrees bit for bit; a held point comes
  back unchanged while a free one moves; a ranged point lands at exactly its
  distance from the reference read at the final pose, for a single image and
  for the mean of two; an infinite distance is reported as a direction; the
  storage decision promotes a marked near point and reports itself by default,
  and is `None` with the crossing off;
  and every rejection above raises `ValueError`.
- **What the crossing changes in the kernel's other tests**: the tests of a
  marked direction held for the whole solve (its frozen translation, the trim
  and protection of its observations) state `FreePointPolicy::NO_CROSS`, since
  under the default a marked free point is solved in inverse depth and decided
  like any other. The `min_track` tests run under the default: a track the trim
  starves comes back bit-identical, a position rather than a bearing through
  its junk rays, and is the one point the decision counts as `unscored`. On
  the noisy low-parallax scene where far direction tracks lock the rotations
  for a focal release, marked directions recover the focal to within 5 px and
  the same tracks crossing to about 8 px, against 76 px off with no far
  tracks. A perturbed finite scene recovers its relative poses either way, and
  the absolute gauge drifts further under the crossing (camera 0 turns by
  about 7e-3 rad), so that test reads poses relative to camera 0.
- **Kept observations only**: a far track with one observation shifted 20 px
  across the baseline, which the final round trims, is stored as the same
  bearing, to 1e-6 rad in the first camera's frame, as the track with that
  observation left out; read over every observation the bearing moves about
  7e-3 rad. Shifted 100 px, the observation would turn the verdict finite and
  does not.
- **Re-estimation under the crossing**: each of the three starting states is
  needed. Without the state the last round left, the midpoint of a far track's
  rays, through a trimmed outlier, starts it near the cameras and the trim then
  drops the whole track; without the mean ray, a direction re-estimated from no
  estimate starts at the midpoint of parallel rays and stays there;
  re-estimating in the representation the round left instead leaves near
  points handed in as directions, and a far direction on a wide fisheye, trimmed
  in every round. A position whose mean ray costs least but reprojects only
  one of five observations under the trim stays a position, and becomes the
  mean ray with a `min_track` of one; a direction whose midpoint, a position,
  does the same stays a direction. The cap is the trim itself: on a track
  with one outlier, a cap of a quarter of the trim, of 1 px, of four times the
  trim or no cap at all takes the midpoint over the state the last round left,
  and the trim keeps that state.
- **Unscored points**: a point the decision cannot score that the rounds left
  in the other representation comes back as it was handed in, a position or a
  direction; one handed in with no estimate keeps the re-estimate it was given
  and is counted in neither `to_finite` nor `to_direction`. The progress line
  appends `, 1 not scored: too few kept observations` for a starved track and
  says nothing of unscored points on a clean solve.

## Protected observations

Appearance-verified observations (e.g. photometric LOO-ZNCC consensus) can
carry corrective long-range signal on a drifted reconstruction — but they
are a 1–2% minority whose *large* residuals are exactly what the staged trim
classifies as outliers, so the unprotected BA silently removes the
correction and re-converges inside the drift gauge. The `protected` mask
lets the caller mark observations whose evidential standing exceeds SIFT
matches so the trim never discards them.

### State and inputs

- A per-observation mask `protected: Option<&[bool]>` (`n_obs`, parallel to
  the observation arrays) marks protected observations. Absent (binding:
  `protected=None`) or all-`false` reproduces the unprotected kernel bit
  for bit.
- A scale multiplier `protected_loss_scale` (default 3.0; the binding
  requires it positive and finite) widens the robust loss for protected
  observations only.

### Semantics

- **Never trimmed.** A protected observation bypasses the inter-round trim
  gates entirely: it stays in the solve set every round regardless of its
  residual, depth, or validity (an invalid protected observation — `NaN`
  point, behind-camera, out-of-domain — contributes the standard penalized
  `(1e6, 0)` residual with a zero Jacobian row: penalized, never steering).
- **Counts toward `min_track`.** Protected observations count as trim
  survivors for their track — they can keep an otherwise-starved track (and
  its unprotected survivors) in the solve — and are themselves never
  dropped by the `min_track` gate. They count toward the `min_obs`
  degenerate-exit floor like any kept observation.
- **Wider robust scale, bounded pull.** A protected observation passes
  through the same soft-L1 loss at scale
  `protected_loss_scale · loss_scale` for the round. Soft-L1's influence
  saturates (the per-component gradient is bounded by `2·s`), so protected
  observations pull with bounded influence rather than being either trimmed
  or dominating a well-supported fit. The widened scale is applied per
  observation inside the solve (cost and Triggs weighting); nothing else in
  the LM changes. Note the saturated cost is still *linear* in the residual
  — protection states that the caller trusts the observation, and does not
  bound the harm a wrong one does: enough mutually inconsistent
  protected pixels can outweigh the clean majority's fit, so the caller
  marks only observations whose evidential standing warrants exactly that
  trade.
- **Re-estimation.** Protected observations participate in the inter-round
  retriangulation / direction re-estimation like any retained observation
  (retriangulation already consumes every supplied observation).
- **Composable with `point_at_infinity`.** The masks are independent — a
  protected direction observation is legal — and both simply apply; there
  is no special casing.

### Binding

`bundle_adjust(..., protected=None, protected_loss_scale=3.0)` — optional
`(n_obs,)` bool array plus the widening multiplier. All other shapes,
validation, and outputs are unchanged.

### Testing requirements (additional)

- **Regression**: an all-`false` mask and an absent mask both reproduce the
  unprotected output bit for bit — with no direction marked and with a
  points-at-infinity mask in play.
- **Trim survival**: a track corrupted with large mutually inconsistent
  offsets is fully trimmed when unprotected (its point passes through
  bit-identical under a single-round schedule) and stays in the solve when
  protected — including through every round of the multi-round default
  schedule — while the clean majority still fits and the junk never gets
  driven to fit (bounded influence).
- **`min_track` interaction**: protected survivors keep an
  otherwise-starved track (and its clean member) in the solve.
- **Gauge correction (load-bearing)**: two internally rigid fragments tied
  only by a ~2% minority of long-range tracks, the second fragment drifted
  by a similarity that leaves every local observation self-consistent.
  Unprotected, the BA trims the long-range observations and is a fixpoint
  of the drift gauge; protected, the same solve recovers the true relative
  gauge (similarity-aligned camera-center RMS, asserted with margin, not
  bitwise). At least three non-collinear shared points are required — two
  leave a 1-DOF family that fits every observation without fixing the
  gauge.
- **Composability**: a protected corrupted direction observation survives
  the trim and pulls its direction, where the unprotected one is trimmed
  (smoke).

## Non-goals

- Rigs. Every image keeps its own free pose; a rig-aware adjustment would solve
  one pose per frame plus camera-to-rig offsets, which is a different
  parameterization.
- Per-observation camera models: an observation's camera is its image's.
- Optimizing distortion beyond the `k1` of `SIMPLE_RADIAL_FISHEYE` and the
  spline of `SFMTOOL_FISHEYE` / `SFMTOOL_PINHOLE`; `opt_k1`/`opt_bspline`
  cover those two radial releases only. Each image's focal and principal point
  are adjusted per image ("The per-image release"); its distortion is always
  its camera's, and the kernel models no rolling shutter. A per-image radial
  correction and per-image rolling-shutter rates for `SFMTOOL_PINHOLE` are
  proposed in
  [per-image-lens-corrections.md](../../drafts/per-image-lens-corrections.md).
- Gauge fixing and covariance estimation. The kernel holds no pose fixed to
  set the gauge and reports no uncertainty. A caller that wants a gauge states
  it through the point constraints (held points, or ranged points, which fix
  the scale; see "Point constraints") or aligns the result afterwards.
- Replacing the production solvers (`sfm solve` wraps COLMAP/GLOMAP). The
  kernel runs inside reconstruction growth and rotation initialization, and
  through the reconstruction-level adjustment it backs the viewer's Bundle
  Adjust command, the `bundle_adjust` MCP tool and `sfm xform --bundle-adjust`
  on spline cameras.

## Open questions

- **The per-image prior's floor and passes.** The floor of `0.001` on a derived
  width is provisional, and so is whether deriving between rounds is enough or
  the final round should run once more under widths derived from its own
  solution. Both are to be checked on solves noisier than the hand-curated
  ground truths, which understate an ordinary solve's noise and spread: Kerry
  Park (the first case with a rig and a wide field, not yet measured) and the
  raw output of `sfm solve` on `dino_dog_toy` and `seoul_bull_sculpture` among
  them, with a script under `scripts/` that repeats the held-out measurement of
  [`bundle-adjustment-measurements.md`](bundle-adjustment-measurements.md#per-image-intrinsic-adjustments)
  and compares the derived widths with the held-out error at fixed widths
  around them.
- **A per-image focal on a two-focal model.** The log focal scale's column
  `(u − cx_i, v − cy_i)` is exact on a two-focal model too, since one
  scale moves both focals; whether to release it past `focal_is_releasable` is
  open.
- **A focal driven by the subject distance.** On `dino_dog_toy` the per-image
  focal scale follows the EXIF `SubjectDistance` closely. A camera could model
  it as `f_i = f₀ + a/D_i`, two shared parameters fitted from every image, in
  place of one free scale per image: a different parameterisation.
- **The dino ground truth.** The `dino_dog_toy` candidate ground truth's spline
  was fitted with one shared principal point and focal, and has absorbed part of
  what the per-image adjustments model. Rebuilding it with them, and refitting
  its spline, is not yet done.
- **The conditioning of a released principal point.** On a narrow field the
  data barely tell a principal-point shift from a turn of every image, and the
  reconstruction-level report states only how far the principal point moved.
  Whether it should carry the solved principal point's standard deviation, read
  from the final linearization's reduced system with the gauge's null space
  projected out (the system is singular along the gauge, so this needs a
  pseudo-inverse), so that a narrow-field solve can say its principal point is
  not determined, is open; covariance estimation is otherwise a non-goal.
- **A damped distortion step beside the held one.** The held-distortion retry
  pins the distortion outright, so when the full step crosses the guard by
  overshooting rather than because the optimum lies past the limit, the
  distortion does not move for the round (the `[(50, 5)]` round run alone in
  the measurements). The alternative is to damp the inadmissible
  camera's distortion slots more, by their own factor on the diagonal, and
  retry at the same `λ` with that factor rising until the step is admissible,
  pinning being the limit. The poses and points would keep the step `λ`
  allows them while the distortion takes the largest admissible damped step.
  It costs more LU solves per rung and a second damping factor to carry
  between iterations. Whether a real case needs it is open, since the
  default schedule's later rounds move the spline anyway.
