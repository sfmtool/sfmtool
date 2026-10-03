# Staged bundle adjustment

## Purpose

The staged robust bundle adjustment written for the cluster pinhole bootstrap
experiments, whose scripts have since been removed: given
images taken through one or more cameras, camera poses, world points, and pixel
observations tying them together, jointly refine the poses and points (and
optionally each camera's focal length and its distortion release, a radial
coefficient or a spline) by minimizing
robust pixel reprojection error over a trim schedule with inter-round retriangulation.
Each camera keeps its own lens parameters in the solve, and every observation is
read through the camera of its own image.

This is the optimizer that the trimmed pose-only refinement
(`crates/sfmtool-core/src/geometry/pose_refine.rs`) is the single-pose
special case of. It replaces the experiment scripts'
`scipy.optimize.least_squares` BA, whose Python-side residual and sparsity
handling dominated the bootstrap's wall-clock.

## Definitions

- `n_cam` **cameras**, each a `CameraIntrinsics` with its own model,
  parameters and initial focal. A camera here is what the `.sfmr` calls one:
  an entry of the camera table, which images reference by index. Rigs are not
  modelled: every image keeps its own free pose.
- `n_img` **images**, each taken through one of the cameras (`image_camera[i]`)
  and each with a
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
`PointConstraint`, `PointConstraints`, `DistanceReference`, `FreePointPolicy`,
`RoundCrossing`), bound as `sfmtool._sfmtool.geometry.bundle_adjust`.

```rust
pub struct BaSchedule {
    pub trim_px: f64,     // pre-round trim threshold on the residual norm
    pub loss_scale: f64,  // soft-L1 scale for the round's solve, px
}

/// The cameras of a solve, and which of them took each image.
pub struct BaCameras<'a> {
    pub cameras: &'a [CameraIntrinsics], // n_cam, each with its initial focal
    pub image_camera: Cow<'a, [u32]>,    // n_img, an index into `cameras`
    pub releases: Option<&'a [CameraRelease]>, // n_cam; None = the flags alone
}

/// What one camera may release; `Default` and `HELD` release nothing.
pub struct CameraRelease {
    pub focal: bool,
    pub distortion: bool, // k1 or the spline, whichever the model carries
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
    free_points: FreePointPolicy,        // whether free points cross representations
    protected: Option<&[bool]>,          // n_obs
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
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
    pub crossings: Vec<RoundCrossing>,   // one per crossing re-estimation
}
```

A single-camera caller passes `&BaCameras::shared(&cam, quats.len())`:

```rust
let out = bundle_adjust(
    &BaCameras::shared(&cam, quats.len()),
    &mut quats, &mut trans, &mut points, &uv, &obs_img, &obs_pt,
    None, None, FreePointPolicy::default(), None, DEFAULT_PROTECTED_LOSS_SCALE,
    true, false, false, &DEFAULT_SCHEDULE, 60, 2, 12, &Progress::none(),
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

**Why `releases` narrows the flags rather than replacing them.** The three
flags are the kernel's whole-solve switches, and every caller that has one
decision for every camera keeps passing them as it always has, with `releases:
None`. A caller that decides camera by camera -- the reconstruction-level
adjustment, whose caller states a release per camera -- passes every flag on
and one `CameraRelease` per camera, and camera `j` releases its focal under
`opt_f && releases[j].focal`, its `k1` under `opt_k1 && releases[j].distortion`
and its spline under `opt_bspline && releases[j].distortion`, each still only
where its model admits it. A list whose length is not `n_cam` is a caller error
and panics with the other shape checks.

**Why the result returns cameras.** A caller wants each solved camera back to
write into its own table. The result carries the whole `CameraIntrinsics` per
input camera, in order: the input camera with its released parameters replaced
by the solved ones (the focal under `opt_f`, `k1` under `opt_k1`, the spline
under `opt_bspline`, where its model admits the release), and equal to the input
camera otherwise. A caller copies it rather than reassembling it from scalars.

`progress` is where the rounds and the LM iterations inside them are reported: a
phase per schedule round with an iteration count under it, so a caller knows
which round a long solve is in. It is also how the solve is asked to stop, which
it is between rounds and between iterations, those being the points where the
state is a whole answer rather than a half-written candidate. A stopped solve
returns the state it had reached rather than an error, since it has no `Result`
to put one in, and the caller asks `Progress::is_cancelled` again to find out.
`&Progress::none()` reports nothing and never stops: every method on it is a
branch on a null sink, and the solve runs exactly as it did before the parameter
existed, which is asserted bit for bit
([../../gui/operation-progress.md](../../gui/operation-progress.md)).

Per schedule round, mirroring the experiment scripts exactly:

1. **Retriangulate (rounds after the first).** Rebuild *every* point from
   *all* supplied observations at the current poses, through the point
   estimation operation
   ([triangulation-rules.md](../reconstruction/triangulation-rules.md)) with `marks`
   on for the round's direction mask, `few = absent`, and the floor, likelihood,
   cheirality and bar rules off, the settings a free point crossing
   representations moves off, and a ranged or held point never reads (see "Point constraints"):
   world rays `R_iᵀ · pixel_to_ray(uv)`, each through the camera of its own
   image, so a track seen by two cameras back-projects each observation through
   its own lens, and centers `−R_iᵀ t_i` per
   observation, grouped by point with a STABLE sort so a track
   accumulates its own observations in the order the caller listed them, and
   solved through [`reconstruction::triangulation::triangulate_batch`]. A track
   with fewer than 2 usable observations becomes `NaN`; a point with no
   observations at all becomes `NaN` too (the callers refill from their full observation set —
   the "refill after BA" rule of the bootstrap spec). Re-admission is the
   point: observations a bad init lost re-enter once the refined cameras
   explain them.
2. **Trim.** Keep observations with residual norm `< trim_px`, an in-front
   measure `> 1e-3 · f`, and a finite point; then drop observations of points
   with fewer than `min_track` survivors. `f` is the focal of the observation's
   own camera at the round's state (the model's first where it carries two). The
   in-front measure is model-aware: the canonical depth `−z_cam` for the
   perspective family, whose projection is defined only for `z_cam < 0`, and the
   range `‖p_cam‖` for a ray-path model (fisheye, equirectangular), which images
   rays out past `θ = 90°`; there the floor rejects only a point on the camera
   centre and the domain test is `ray_to_pixel`, whose failure is an invalid
   residual. A direction's measure is checked against zero (see "Points at
   infinity").
   If fewer than `min_obs` observations survive, return degenerate: state
   passes through, `residual_norms` all `+∞` (the fast bootstrap's
   "wildly wrong focal" guard).
3. **Solve.** One robust sparse Levenberg–Marquardt solve (below) over the
   kept observations at the round's `loss_scale`.

After the last round, `residual_norms` is the unweighted reprojection
residual norm of **every supplied observation** at the final state (`+∞`
where invalid), so callers tally inlier fractions against denominators of
their own choosing.

## The solve

Levenberg–Marquardt over a local parameterization, minimizing the soft-L1
robust cost applied per residual COMPONENT (matching scipy's element-wise
`loss="soft_l1"` that this kernel replaces)

```
cost = Σ_i s² · ρ(r_i² / s²),   ρ(z) = 2·(√(1 + z) − 1),   s = loss_scale
```

- **Parameters.** Per touched image a local `SO(3) × ℝ³` perturbation
  (`R ← exp(δθ)·R`, `t ← t + δt`); per touched point `X ← X + δX`; and per
  camera, when `opt_f`, its focal `f_j ← f_j + δf_j`; when `opt_k1`, its
  radial coefficient `k1_j ← k1_j + δk1_j`; when `opt_bspline`, its
  spline `c_j,i ← c_j,i + δc_j,i`. The three flags are requests to every
  camera, each decided on that camera's own model (below and "Several
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
  fisheye periphery past `θ = 90°` included. `opt_k1` is the fisheye family's radial rung
  and requires `SIMPLE_RADIAL_FISHEYE`:
  `∂(u, v)/∂k1 = f·θ³·(ûx, ûy)` exactly — the θ³ curvature the
  equidistant map cannot express, which is what lets the adjustment
  flatten a lens's residual field instead of buying it back with
  geometry (the finite dome that pulls sky and horizon off infinity).
  Every other model fails the conditions — a second focal `fy` (no slot
  in the camera block), or higher polynomial coefficients recovered
  through `f`-dependent normalization. The binding rejects `opt_f` /
  `opt_k1` for those loudly, and the core silently holds that camera's
  parameter fixed (never a half-modeled DOF) while releasing it on the cameras
  whose models admit it. The rung also needs the
  model's INVERSE to carry `k1`: retriangulation and direction
  re-estimation read `pixel_to_ray`, which for `SIMPLE_RADIAL_FISHEYE` is
  the Newton recovery of `θ` without the family's wide-angle blend — that
  blend hands back the identity `θ = r_d` ray past 90°, dropping `k1`
  exactly where `k1·θ³` is largest (a 105° rim at `k1 = 0.02` comes back
  6° off, which is a ray, not a rounding error).
- **The spline release.** `opt_bspline` is the radial rung of
  the two models that carry one, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`
  ([../../formats/sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md)),
  and only when that spline is defined: at least two coefficients on a
  positive finite domain end (`bspline_theta_max` / `bspline_rho_max`;
  anything shorter evaluates as the identity and has nothing to release).
  The released block is the camera's whole coefficient vector `c₀..c_{N−1}`,
  shared by every image that camera took like its `f` and `k1`, so with one
  camera the reduced camera system is `[f, k1, c₀..c_{N−1} | 6·n_im]`, width
  `2 + N + 6·n_im` (see "Several cameras" for more). `opt_k1` and
  `opt_bspline` are mutually exclusive on any one camera: no model carries both
  parameters, so the binding rejects the combination (checked before the model
  gates, so the caller sees the real reason) and the core degrades each on its
  own model test. Retriangulation and direction re-estimation read the model's Newton
  inverse, which carries the spline over the model's whole radial domain, for
  the same reason the `k1` rung needs its own (for the fisheye that means no
  wide-angle blend; for the pinhole, an explicit Newton arm rather than the
  generic fixed-point undistortion).
  **Unsupported coefficients are pinned.** A coefficient whose basis span no
  surviving observation touches has an exactly-zero column, hence an exactly
  zero `h_cc` diagonal (a sum of squares) and a singular reduced system; the
  same pinning that holds an unreleased `f` or `k1` — and a frozen
  translation — holds it, so it comes back bit for bit as it went in and the
  spline never moves past the data. **Step guard.** Inside the damping
  ladder, exactly where a non-positive focal or a folded `k1` is rejected, a
  candidate spline that is non-finite or violates the model's monotonicity
  invariant `1 + δ'(d) > 0` anywhere on `[0, d_max]` — or on the linear tail
  past it, where the single `1 + δ'(d_max) > 0` decides the whole half-line —
  is rejected and the step
  re-damped. The whole domain rather than the imaged field: monotonicity is
  the model's construction invariant (the bracket behind its inverse) and the
  accepted spline is persisted into the camera, while unsupported
  coefficients are pinned anyway, so the wider check costs no legitimate
  steps.
- **Staged releases.** Callers open the shared parameters in stages:
  fixed → `opt_f` → `opt_f` plus a distortion release, so a distortion
  parameter opens only on a focal that has already settled — `opt_f + opt_k1`
  for the `k1` rung, `opt_f + opt_bspline` for the spline. The spline
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
  has, and refuses the distortion without the focal. That is how the
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
  equirectangular, which have no analytic form — composed with
  `−[R·X]ₓ` (rotation), `I₃` (translation), and `R` (point) blocks,
  exactly as in `pose_refine.rs` (including the fallback). An observation
  whose point is behind the camera / outside the model domain contributes
  residual `(1e6, 0)` with a zero Jacobian row — penalized, never
  steering. The shared-parameter columns are analytic too: `(u − cx)/f`
  for the focal, `f·θ³·(ûx, ûy)` for `k1`, and
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
  block stays fixed-width (`[f, k1, 4 active coefficients, 6 pose]`)
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
  acceptance. First-order IRLS was measurably worse here: its shallower
  valley model stopped the focal release short on seoul (kept f at the
  scan winner where scipy walked −20% to the reference focal).
- **Schur complement.** Points are eliminated: per-point 3×3 blocks are
  inverted directly and the reduced camera system (one lens block
  `[f_j?, k1_j?, c_j,0..c_j,N_j−1?]` per camera, then `6·n_im`, dense) is
  solved by LU; point updates back-substitute. Unreleased lens slots (and
  released coefficient slots with no observation support, and every slot of a
  camera no kept observation reaches) are pinned to an identity row/column with
  a zero gradient entry, which is what keeps that system regular.
  Rejected steps re-damp and re-solve from the same linearization (no
  re-evaluation), with Marquardt scaling `λ·diag(JᵀJ)` for the
  `x_scale="jac"` parameter-scale invariance of the scipy original.
- **Termination.** `max_iters` accepted-step budget per round; stop early
  when accepted steps improve the cost by less than `1e-8` relative TWICE
  in a row (one tiny step is how a traverse of a nearly-flat valley starts
  — the focal release walks −20% through one), or when no damping in a
  bounded ladder (12 ×4 escalations, capped at `λ = 10¹²`) finds a
  downhill step.

## Several cameras

Everything above is stated for one camera, and a solve over several is the same
solve with the lens parameters kept per camera. `BaCameras` carries the camera
list and, per image, the index of the camera that took it.

### The reduced camera system

The reduced system has one lens block per camera, then the poses:

```
[ f₀?, k1₀?, c₀,₀..c₀,N₀−1? | f₁?, k1₁?, c₁,₀..c₁,N₁−1? | … | 6·n_im ]
```

Camera `j`'s block has a focal slot, a `k1` slot and `N_j` spline slots, where
`N_j` is the length of its spline when its spline is released and `0`
otherwise. The system's width is `Σ_j (2 + N_j) + 6·n_im`. An observation of image `i`
writes its lens columns into the block of camera `image_camera[i]` and into no
other camera's block, so two cameras' lens parameters couple only through the
poses and points they both observe. A few cameras add a few slots beside the
`6·n_im` pose slots, so the dense solve costs what it does with one.

The per-observation camera block keeps its fixed width. The spline
instantiation (`2 + 4 + 6` columns) is used whenever any camera releases a
spline; in it, a spline column that carries no coefficient (one of the
gauge-anchored pair, or any spline column of a camera whose spline is not
released) points at the observing camera's own `k1` slot and is exactly zero,
so it adds exact zeros there whether or not that slot is released.

Every rule the single block follows applies to each block separately:

- **Release gates, per camera.** `opt_f`, `opt_k1` and `opt_bspline` are
  requests to every camera, narrowed per camera by `BaCameras::releases` where
  it is given. A camera whose model the release is not exact for keeps that
  parameter fixed, so a `SIMPLE_PINHOLE` and an `OPENCV_FISHEYE` in one solve
  under `opt_f` release the pinhole's focal and hold the fisheye's, and a camera
  whose entry releases nothing is held whatever the flags say.
- **Pinning.** A slot that is not released is pinned with an identity row and
  column and a zero gradient entry, and so is every slot of a camera none of
  whose images has a kept observation in the round. Such a camera is not
  stepped, and comes back exactly as it went in.
- **Step guards.** A candidate step is rejected, and re-damped, when any
  camera's candidate focal is non-positive, its `k1` folds inside its own imaged
  field (the largest pixel radius of its own kept observations about its own
  principal point), or its spline breaks monotonicity on its own domain.
- **Staged releases** are the caller's schedule, as with one camera.

### What reads the camera, per observation

Everything that reads a camera reads the camera of the observation's image,
`cameras[image_camera[obs_img[k]]]`: the projection, its Jacobian and the lens
columns; the trim's in-front measure and its `1e-3 · f` floor; and
`pixel_to_ray` in the inter-round re-estimation.

Under `FreePointPolicy::cross` the crossing reads each camera twice more. The
noise level pools every camera's residuals and gates each camera's against its
own robust spread, as the stored measure does, and each ray's noise weight is
the projection derivative of its own image's camera, so a track seen through
two lenses weights each sighting by the pixels per radian of the lens that took
it.

### Parity

With one camera and every image on it, the layout above reduces to
`[f, k1, c₀..c_{N−1} | 6·n_im]` and every per-camera read reads that camera, so
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
                               # world-frame direction. None/all-False
                               # reproduces the finite-only kernel bit for bit
    held=None,                 # (n_pt,) bool; a held point's coordinate is the
                               # caller's for the whole solve
    distance=None,             # (n_pt,) float64; a ranged point's distance,
                               # +inf for a direction, NaN where not ranged
    distance_from=None,        # (n_pt,) image index, or a sequence of image
                               # indices to average, with -1 where there is
                               # none; a finite `distance` requires one
    free_points_cross=False,   # re-decide every free point's representation at
                               # each inter-round re-estimation, on the
                               # point-or-bearing test at the measured noise
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
    schedule=[(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)],
    max_iters=60,
    min_track=2,
    min_obs=12,
) -> dict                      # cameras (list of CameraIntrinsics),
                               # quaternions_wxyz (n_img, 4),
                               # translations (n_img, 3), points (n_pt, 3),
                               # residual_norms (n_obs,),
                               # point_at_infinity (n_pt,),
                               # crossings (list of dict)
```

`cameras` in the result holds one `CameraIntrinsics` per input camera, in
order: the kernel's returned cameras, each the input camera with its released
parameters replaced by the solved ones. A caller reads a solved focal as
`out["cameras"][j].focal_lengths[0]` and a solved `k1` or spline from the
camera's `parameters`.

The binding holds every camera to every release it is asked for: a release that
some camera's model does not admit raises a `ValueError` naming that camera,
rather than letting the kernel hold that camera's parameter fixed. `opt_bspline`
raises for a camera that is neither `SFMTOOL_FISHEYE` nor `SFMTOOL_PINHOLE`, and
for an undefined spline; `opt_k1` together with `opt_bspline` raises first, so
the caller sees the exclusion rather than whichever model gate happens to fire.
An empty `cameras`, an `image_camera` whose length is not `n_img`, and an index
past `cameras` raise `ValueError` as well.

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
and is how a caller learns the outcome of a crossing: `True` where the returned
row is a direction, `False` where it is a position. `crossings` is
`BundleAdjustment::crossings` as one dict per re-estimation, with the
`RoundCrossing` fields as keys (`round`, `sigma_px` or `None`,
`observation_count`, `outlier_count`, `to_finite`, `to_direction`), and empty
without `free_points_cross`.

A reconstruction read from a `.sfmr` carries its constraints as the
`point_constraints` column, a `uint8` array in the canonical numbering (`0`
free, `1` ranged, `2` held) whatever legend the file stored, and `write_sfmr`
takes it the same way; the file's legend never reaches the dict or
`SfmrReconstruction.point_constraints`.
`sfmtool._sfmtool.io.POINT_CONSTRAINT_NAMES` is that canonical numbering as a
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
- **The curvature rung**: `∂(u, v)/∂k1` matches a central difference of the
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
  unreleased coefficients bit for bit; coefficient slots with no observation
  support hold nonzero input sentinels exactly through a field-limited
  scene; and directions carry the rung where a near-axis finite cloud
  cannot.
- **The spline step guard**: the admissibility predicate rejects a folded
  spline and non-finite coefficients on either radial coordinate; end to
  end, a released solve never returns a spline violating `1 + δ'(d) > 0` on
  `[0, d_max]`.
- **Directions carry the rung**: on a scene whose finite cloud sits near the
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
  - *The crossing reads each lens.* A crossing round over images split between
    two focal lengths measures the level the residuals of both cameras give, and
    decides each track as `bearing_score` does on rays weighted through the
    camera of each sighting's image.
  - *Shape checks.* An index past the camera list, an `image_camera` of the
    wrong length and an empty list panic; the binding raises `ValueError` for
    each, and names the camera a refused release does not admit.

## Points at infinity

There is one staged loop, not two: direction handling is a per-point branch
inside it, so with nothing marked the loop *is* the finite-only solve. (It was
originally a second mirrored copy of the whole kernel, entered only when a
direction was marked; the copies were merged once the reduction was shown to
hold bit for bit.)

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
  model-aware test above, against a floor of zero instead of `1e-3 · f`.
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

- **Free.** The solve owns the point. Its representation is whatever its rays
  support at the current geometry, decided by the re-estimation between rounds
  under `FreePointPolicy` and allowed to change in either direction as the
  poses move.
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
              true, false, false, &DEFAULT_SCHEDULE, 60, 2, 12);
```

An absent `constraints` is every point free, and with a default
`FreePointPolicy` (`cross = false`) the kernel is the one the sections above
describe, bit for bit. The Rust interface is
[bundle_adjust.rs](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs)
(`PointConstraint`, `PointConstraints`, `DistanceReference`, `FreePointPolicy`);
the Python binding takes the same three constraints as flat per-point arrays
(see [Bindings](#bindings)), and a reconstruction carries them in the `.sfmr`
constraint triple (see
[Per-point constraints](../../formats/sfmr-file-format.md#per-point-constraints-optional-version-7)).

### Free points: crossing between representations

A proposal to solve free points in inverse depth, so that a point can move
between near and infinity within a round, is the amendment draft
[point-or-bearing-likelihood-ratio.md](../../drafts/point-or-bearing-likelihood-ratio.md).

```rust
#[derive(Default)]
pub struct FreePointPolicy {
    pub cross: bool, // re-decide free points between rounds; off by default
}

pub struct RoundCrossing {
    pub round: usize,            // the round the re-estimation opened (1, 2, ...)
    pub sigma_px: Option<f64>,   // the level it decided at; None: nothing crossed
    pub observation_count: usize,
    pub outlier_count: usize,
    pub to_finite: usize,        // free directions that came out finite
    pub to_direction: usize,     // free points that came out directions
}
```

Within a round nothing changes: a finite point perturbs in three Euclidean
degrees of freedom and a direction in the two of its tangent plane. The crossing
happens where the representation is already re-read, the inter-round
re-estimation, which under `cross` runs the retriangulation operation with
`marks` **off** for free points, `cheirality` **on**, `few = absent`, and the
`likelihood` rule at the round's measured noise level `σ`
([triangulation-rules.md](../reconstruction/triangulation-rules.md) § "The
likelihood rule"). Each free track's rays and noise weights are built through
the camera of each observing image as it stands at the round, with the focal,
`k1` or spline a release has walked to, and the track is decided by the
point-or-bearing test
([batch-triangulation-api.md](../reconstruction/batch-triangulation-api.md)
§ "Point or bearing"): a track whose rays ask for a depth at `σ` is finite, one
whose rays do not becomes the bearing the test fits, and both are ordinary
outcomes rather than events. The level comes from the previous round's kept
observations only, but the verdict reads every observation of the track at the
round's geometry, those the trim left out included, as the re-estimation always
has. A track that turns finite starts from the
midpoint the operation solves, and the next round's solve places it, so no
separate point fit runs. A track whose fitted bearing is behind an observing
camera is not stored as that bearing: it goes to the solve, and a solved point
behind a camera becomes the mean ray under `cheirality`, as a thin track did
before the test.

The verdict is written into the mask the next linearization reads; the first
round has no re-estimation, so the caller's input mask is what the first
linearization uses. A track whose estimate comes back absent keeps the
representation it had, so momentarily losing its observations does not also
change its constraint. Ranged and held points do not cross. A free point that
ends as a direction comes back as a unit row, as any direction does.

**The noise level is measured each round.** `σ` is the RMS per-axis residual
over the observations the previous round solved on (its kept set, after the
trim and `min_track`) of points that were finite in it, at the state that
round's solve settled on: the estimator of the stored measure
([batch-triangulation-api.md](../reconstruction/batch-triangulation-api.md)
§ "The measured noise level"), each camera's residuals gated at
`OUTLIER_GATE` robust spreads, no degrees-of-freedom correction, and never
under the cameras' keypoint resolution. A direction's residuals are left out,
because they are the bearing's, which is the model in question.

- **Not the loss scale.** `loss_scale` is a schedule constant, chosen before
  anything is measured, and one schedule runs over captures whose noise
  differs fivefold: the measured levels of the clean inputs below run from
  0.19 px on a Kerry Park solve to 1.06 px on the 85-image `dino_dog_toy`
  solve. A level read off the residuals follows the scene being adjusted.
- **From the last round's kept set.** Those are the residuals the solve has
  just minimised, so the level follows the adjustment as it converges. The
  trim before that round capped every residual it kept at its `trim_px`
  (50 px, then 12, by default), which bounds what a gross outlier can add once
  the poses are rough enough that the robust spread grows with them and the
  gate passes it (the stored measure's known limitation).
- **A round with no finite observation** has no level. It decides nothing:
  every free point keeps the representation it had, and its `RoundCrossing`
  reports `sigma_px: None`.
- **One more projection per round.** The residuals are projected once, at the
  top of each crossing round, over the previous round's kept observations. The
  solve's own evaluations are of candidate states, and keeping the accepted
  one's residuals would mean carrying them through every damping step for a
  pass that costs less than one iteration. On the 36,587 observations of the
  `dino_dog_toy` solve, the default schedule takes 4.93 s with the crossing on
  and 4.91 s with it off.

**A rough round decides more bearings.** A round whose poses are far from
converged leaves large residuals and measures a large `σ`, and at a large `σ`
fewer tracks ask for a depth. As the rounds converge `σ` falls toward the
capture's noise and the tracks whose rays carry a depth at that level come back
finite. With five iterations a round, so that the first rounds stop short of
convergence, from poses perturbed by 1° and 5% of the camera extents and 1 px
of keypoint noise, the first re-estimation measures 6.0, 5.1 and 7.2 px on
`tk117`, the seoul bull ground truth and a seoul bull `sift_files` solve, and
the second 2.9, 2.7 and 3.5 px; the first makes 54, 64 and 232 points
directions and the second makes 23, 54 and 157 finite again. With the default
60 iterations the first round gets close enough from the same start (on
`tk117` it uses all 60 without meeting its convergence test) that the first
re-estimation measures the noise: 0.91, 0.90 and 0.79 px.

On the Kerry Park solve the same five-iteration start makes most of the
reconstruction directions and leaves it worse than not crossing at all. Its
first round stops with `σ` at 3.1 px where the converged level is about
0.8 px, so the level is mostly pose error. That capture is a short-baseline
rig with about 3.6 observations per point, and at 3 px 71% of its points show
no depth, against 5 to 14% of the other three inputs' points at the same level
(whose own first re-estimations, at their 5 to 7 px, make 14 to 28% of their
points directions); the first re-estimation makes 633 of its 886 points
directions, although their rays do
carry a depth at the converged level. Solved as directions in the next round,
they bend the poses: that round's finite `σ` rises from 2.67 to 3.26 px,
where without the crossing it falls from 3.11 to 2.23. The second
re-estimation measures that worse state over only the 1,008 observations of
the points still finite and keeps them directions (4 back, 29 more), and the
solve ends with 658 directions and a median residual of 2.03 px, against
1.05 px with the crossing off. Recovery needs the level to fall: with eight
rounds the third re-estimation measures 1.03 px and makes 359 of them finite
again. This happens only when a round stops on its iteration budget far from
convergence; on that solve it takes ten iterations a round or fewer, and with
the default 60 no start up to 4° and 20% of the camera extents makes it, nor
does any production caller cross free points (every one passes
`FreePointPolicy::default()`).

The crossing has no guard against it. Letting a round that used its whole
iteration budget only promote, since an inflated level errs toward demotion
only, fixes this case (no directions, median residual 1.05 px), but it also
holds back legitimate demotions after a default-budget round that used its 60
iterations: with poses perturbed by 0.2° and 1% and 1 px of keypoint noise,
the Kerry Park solve's disagreements with the test at its own result rise
from 34 to 67 (its 150 demotions wait for the last re-estimation) and
`tk117`'s from 0 to 1. Skipping the crossing after such a round regresses the
same case, and reverting a re-estimation's demotions past a share of the free
points depends on a share that a legitimate first re-estimation on that solve
already reaches (18%). A solve whose rounds converge is the case the crossing
is for; solving free points in inverse depth within each round (the amendment
draft) removes the dependence on the round boundary.

**No damping.** The rule has no hysteresis, because none was needed. The
round's solve fits the representation it was handed, and that already pulls a
borderline track's score toward the side it is on: over ten cameras and 300
points with 0.76 px of noise, a track 1,500 units out whose rays score 33 at
the true poses scores 38 at the end of a solve that started it finite and 6 at
the end of one that started it as a direction, and stays as it started. Nothing in the
measurements flips back and forth. Over the seven re-estimations of an
eight-round schedule on a Kerry Park solve with 1 px of keypoint noise, the
points changing at each are 137, 39, 26, 14, 7, 6 and 7 as marginal far points
settle as directions; over the eight re-estimations of a nine-round schedule
3 of its 886 points change twice (each once each way, rounds apart) and none more; in the synthetic sweep
of the testing requirements, a track crosses at the first re-estimation or not
at all.

**Measured against the floor.** The default schedule over the four inputs of
the point-or-bearing work, clean and degraded (poses perturbed by 0.2° and 1%
of the camera extents, 1 px of keypoint noise, both, and 1° and 5% with
1 px), with every free point crossing and the focal held. The floor is the
rule this one replaced, `θ_floor = 2 · loss_scale / f`. "Disagree" counts the
points whose stored representation the point-or-bearing test at the result's
own measured noise disagrees with, which is what `sfm analyze
--depth-reliability` lists; "dirs" are the directions in and out.

| Input | Start | Floor: dirs, disagree | Likelihood: dirs, disagree | Median residual, floor / likelihood (px) |
|---|---|---|---|---|
| `tk117` | clean | 12 → 11, 2 | 12 → 9, 0 | 0.169 / 0.168 |
| `tk117` | 1 px keypoints | 12 → 2, 10 | 12 → 12, 0 | 1.025 / 1.018 |
| `tk117` | 1° and 5%, 1 px | 12 → 2, 13 | 12 → 12, 0 | 1.049 / 1.015 |
| seoul bull ground truth | clean | 14 → 14, 0 | 14 → 14, 0 | 0.256 / 0.256 |
| seoul bull ground truth | 0.2° and 1%, 1 px | 14 → 12, 2 | 14 → 14, 0 | 0.904 / 0.911 |
| Kerry Park solve | clean | 0 → 45, 23 | 0 → 0, 0 | 0.190 / 0.173 |
| Kerry Park solve | 1 px keypoints | 0 → 17, 163 | 0 → 174, 27 | 0.791 / 0.863 |
| Kerry Park solve | 1° and 5%, 1 px | 0 → 17, 183 | 0 → 218, 33 | 0.781 / 0.897 |
| seoul bull `sift_files` solve | every start | 0 → 0, 0 | 0 → 0, 0 | equal |

On `tk117` the test makes points 298, 294 and 295 finite, as reclassification
does, where the floor promoted one; at 1 px of keypoint noise the floor, whose
angle does not move with the noise, promotes ten of the twelve bearings and the
test none. On the clean Kerry Park solve the floor demotes 45 points, 23 of
which the test calls finite. The 27 to 33 disagreements left on the noisy Kerry
Park solve are marginal far points still settling as directions when the three
rounds end (eight rounds leave 8), and the median residual rises with them
because a bearing explains a marginal point's pixels slightly worse than a
point does. No input produced a `NaN` point, and the count of observations
under 4 px is within one of the floor's, or higher, on every input.

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
  observations, an absent `constraints` and an all-free one under a default
  policy agree on every output field to the bit (poses, points, focal,
  residual norms and the reported representation), and neither reports a
  crossing round.
- **The ranged Jacobian**: the analytic blocks of every observation, assembled
  into a dense Jacobian, match a central difference of the whole residual
  vector on a small ranged scene, with the reference image apart from the
  observers, among them, and as the mean of two.
- **Crossing, both directions**: a near cloud started at infinity comes back
  finite at its true positions, and a far track started finite in a capture
  with a third of a pixel of noise comes back as a unit direction, with the
  reported representation matching the row and one crossing counted. On exact
  pixels the measured level is the keypoint resolution and any parallax is a
  depth, so the far case needs the noise.
- **The boundary is the measured noise**: over ten cameras and 300 points, the
  same track 3,000 units out is finite with 0.09 px of noise and a bearing with
  0.76 px, from either starting representation, and the measured level follows
  the noise put in.
- **A crossing round, by hand**: with no iterations and no trim, so the state
  the first re-estimation reads is the input, its level is the gated RMS of the
  finite residuals computed by hand (a 40 px mismatch is the one outlier), and
  every verdict and bearing is that of `bearing_score` and `is_finite` on rays
  `observed_ray` builds through each image's own camera, over two cameras of
  different focal length.
- **No finite observation, no crossing**: a directions-only solve reports no
  level and changes no representation.
- **No oscillation**: across distances whose score runs from about 60 to about
  8 at the measured level, from both starting representations, over six rounds,
  a far track crosses at the first re-estimation or not at all.
- **A rough start**: from rotations perturbed by up to 0.6° and two
  iterations a round, the first re-estimation measures more than 1.5 times the
  level of the last, every near point ends finite, the far tracks end
  directions, nothing is `NaN`, and the last re-estimation changes nothing.
- **An unconverged round demotes, and later rounds restore**: from rotations
  perturbed by up to about 6° about each axis, translations by up to 0.2 units
  along each axis, and three iterations a round, the first re-estimation
  measures more than five times the level of the last and makes at least 25
  points directions (its `to_direction`), which the thirty points 250 to 395
  units out supply, though their rays carry a depth at the converged level;
  the rounds after it make them finite again, and every one ends finite. With
  60 iterations a round no re-estimation makes any point a direction. A fitted
  bearing behind a camera and a rescue's survivors are covered by the rule's
  own tests ([triangulation-rules.md](../reconstruction/triangulation-rules.md)
  § "Testing").
- **Held points**: their coordinates come back to the bit while free points
  move, and their observations still carry residuals; an image whose only
  finite evidence is one held point solves its translation, where the same
  landmark as a direction leaves the translation frozen to the bit.
- **Ranged points**: an infinite distance reproduces a marked direction bit for
  bit; a finite one comes back at exactly its distance from the reference read
  at the final pose; and a landmark started at a wrong bearing but its true
  distance recovers the bearing the reference image sees, where the same track
  free converges to a wrong depth.
- **The binding**: its off position (absent arguments, and explicitly-off ones)
  agrees bit for bit; a held point comes back unchanged while a free one moves;
  a ranged point lands at exactly its distance from the reference read at the
  final pose, for a single image and for the mean of two; an infinite distance
  is reported as a direction; the crossing promotes a marked near point; and
  every rejection above raises `ValueError`.

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
  — protection is vouching, not a safety net: enough mutually inconsistent
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
- Releasing some cameras and not others. The release flags are requests to
  every camera, decided per camera only by what its model admits.
- Per-observation camera models: an observation's camera is its image's.
- Optimizing distortion beyond the `k1` of `SIMPLE_RADIAL_FISHEYE` and the
  spline of `SFMTOOL_FISHEYE` / `SFMTOOL_PINHOLE`, or the principal point;
  `opt_f`/`opt_k1`/`opt_bspline` cover each camera's focal and those two radial
  releases only.
- Gauge fixing, covariance estimation, or constraint handling — callers
  own the gauge (the bootstrap's evaluation aligns by similarity anyway).
- Replacing the production solvers (`sfm solve` wraps COLMAP/GLOMAP); this
  kernel serves the bootstrap experiments and whatever grows out of them.
