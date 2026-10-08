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
its own image.

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
`FreePointDecision`), bound as `sfmtool._sfmtool.geometry.bundle_adjust`.

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
    free_points: FreePointPolicy,        // default: free points cross; NO_CROSS opts out
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
    pub free_point_decision: Option<FreePointDecision>, // the storage decision; None under NO_CROSS or on a degenerate exit
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

Per schedule round:

1. **Retriangulate (rounds after the first).** Rebuild *every* point from
   *all* supplied observations at the current poses, through the point
   estimation operation
   ([triangulation-rules.md](../reconstruction/triangulation-rules.md)) with `marks`
   on for the round's direction mask, `few = absent`, and the floor, likelihood,
   cheirality and bar rules off (under `FreePointPolicy::cross`, `cheirality`
   on, and a free point starts the round from the best of three states, see
   "Re-estimation between rounds" under "The parametrisation"; a ranged or held
   point reads neither, see "Point constraints"):
   world rays `R_iᵀ · pixel_to_ray(uv)`, each through the camera of its own
   image, so a track seen by two cameras back-projects each observation through
   its own lens, and centers `−R_iᵀ t_i` per
   observation, grouped by point with a STABLE sort so a track
   accumulates its own observations in the order the caller listed them, and
   solved through [`reconstruction::triangulation::triangulate_batch`]. A track
   with fewer than 2 usable observations becomes `NaN`; a point with no
   observations at all becomes `NaN` too. What happens to such a point is the
   caller's choice: growth refills it from its full observation set after the
   adjustment, and the reconstruction-level adjustment deletes it.
   Re-admission is the point: observations a bad init lost re-enter once the
   refined cameras explain them.
2. **Trim.** Keep observations with residual norm `< trim_px`, an in-front
   measure over the round's in-front floor (below), and a finite point; then
   drop observations of points with fewer than `min_track` survivors. The
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
  equirectangular, which have no analytic form, with a step of `1e-6` of the
  camera-frame point's range (the projection is homogeneous of degree zero, so
  its derivative scales as one over the range and a step that scales with it
  is equally accurate at every world scale) — composed with
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
columns; the trim's in-front measure; and `pixel_to_ray` in the inter-round
re-estimation. The in-front floor is the scene's, not a camera's, so it reads
no camera.

Under `FreePointPolicy::cross` the storage decision reads each camera twice
more. The noise level pools every camera's residuals and gates each camera's
against its own robust spread, as the stored measure does, and each ray's noise
weight is the projection derivative of its own image's camera, so a track seen
through two lenses weights each sighting by the pixels per radian of the lens
that took it.

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
    schedule=[(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)],
    max_iters=60,
    min_track=2,
    min_obs=12,
) -> dict                      # cameras (list of CameraIntrinsics),
                               # quaternions_wxyz (n_img, 4),
                               # translations (n_img, 3), points (n_pt, 3),
                               # residual_norms (n_obs,),
                               # point_at_infinity (n_pt,),
                               # free_point_decision (dict or None)
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
  - *The storage decision reads each lens.* A decision over images split
    between two focal lengths measures the level the residuals of both cameras
    give, and decides each track as `bearing_score` does on rays weighted
    through the camera of each sighting's image.
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
              true, false, false, &DEFAULT_SCHEDULE, 60, 2, 12, &Progress::none());
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
points could not be scored and `; the final round stopped on its iteration
budget` when it did not converge, or `free points not decided: the final round
kept no observation of a finite point`. An empty schedule, which only measures,
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
[Kerry Park ground truth](../../../test-data/images/kerry_park/kerry_park_ground_truth.sfmr)
with the crossing off, every point held in the representation the decision
stored, lowers its cost by 17.3 noise units and takes the scores of the six
far points the decision made finite (see the table below) back under the
threshold of 25, to 18.8 to 24.2. No refit follows the decision, because a solve
that carries a point in the representation just chosen is the direction bias
described below ("Why inverse depth"), which pulls each score toward the side
it was put on; the poses the inverse-depth solve leaves are the ones fitted with
no representation imposed.

#### Why inverse depth

The alternative is to carry each free point as a position or a direction for a
whole round and re-decide it on the test between rounds, and it is rejected
because the geometry of a round is then solved with the point in the
representation it has, and the solve fits that representation: over ten cameras and 300 points with 0.76 px of noise, a track
1,500 units out whose rays score 33 at the true poses scores 38 at the end of a
solve that starts it finite and 6 at the end of one that starts it as a
direction, so a borderline track keeps the representation it starts with, and
over six cameras and 40 points one round bends the poses by up to half a degree
to fit a far track as whichever it is given. A marginal track needs several
rounds to settle (on a Kerry Park solve with 1 px of keypoint noise the points
changing at each of the seven re-estimations of an eight-round schedule are
137, 39, 26, 14, 7, 6 and 7). And a round that stops on its iteration budget
far from convergence measures a level that is mostly pose error, makes points
with a depth directions at it, and the next round bends the poses to fit them:
on the Kerry Park solve with five iterations a round, from poses perturbed by
1° and 5% of the camera extents and 1 px of keypoint noise, the first
re-estimation measures 3.1 px and makes 633 of its 886 points directions, and
the solve ends with 658 directions and a median residual of 2.03 px, against
1.05 px with the crossing off.

In inverse depth `ρ = 0` is a value the solve can reach, so a point carries no
representation through a round and the poses are not fitted to one chosen
before it; a far
point stays well conditioned because its inverse depth is close to linear in
the observations; and nothing is decided at a round boundary, so a rough round
cannot demote anything. Over 24 far tracks spanning the threshold in each of
four noise draws on the ten-camera scene above (96 tracks, 0.81 px of noise),
the verdict at the result agrees with the verdict at the true poses on 85
tracks solved in inverse depth, on 83 carried as positions and on 74 carried as
directions; the scores read at the result move from those at the true poses by
−7.2 to +13.7 on average per draw in inverse depth and by −11.8 to −13.6 when
carried as directions.

#### Decisions

**Translation observability needs only the pinning directions already had.** For
`ρ > 0` the translation column is the Euclidean point's (`ρ·J` at `p̃ = ρ·p_cam`
is `J` at `p_cam`), and the Schur complement is invariant under an invertible
reparametrisation of a point, so the reduced camera system is the Euclidean one
up to the Marquardt damping. Measured at the first linearization of the first
round, where both parametrisations read the same state, the translation blocks
of the reduced system agree to the printed digits for the images observing the
most far points (on the Kerry Park solve with 1 px of keypoint noise up to 30%
of an image's observations on points more than fifty times farther from their
anchor than the camera is), with smallest-to-largest eigenvalue
ratios of 0.09 to 0.65. Only `ρ = 0` exactly makes the column zero, and that is
the case the existing pin covers; no damping that reads the column's size is
needed.

**The anchor is re-read at the start of every round.** The anchor only chooses
the parametrisation, but the bound at `ρ = 0` and the singularity at the anchor
are places a point cannot move through, and an anchor left at the input poses
can sit away from cameras a rough start has moved. Because the re-estimation
reads a position between rounds anyway, `ρ` keeping its meaning across rounds
buys nothing. Measured on the default schedule, the two choices store the same
representation for every point, and agree on the median residual to 0.001 px,
on every start up to 1° and 5% on the seoul bull ground truth, the Kerry Park
solve and the seoul bull `sift_files` solve. From 4° and 20% on the
Kerry Park solve, at 200 iterations a round, where both converge, the anchor at
the input poses stops the second round after 17 iterations and the result keeps
3,002 observations under 4 px, with 165 directions and 9 points disagreeing with
the test at the converged level; re-anchored, the second round runs 102
iterations further and the result keeps 3,120 (3,016 with the crossing off),
with 147 directions and none disagreeing.

**The decision is read at the end of an unconverged solve too.** A final round
that stops on its iteration budget measures a level that still carries pose
error, which errs toward a direction, and `converged: false` says so. Not
deciding was measured as the alternative, storing each point as the solve left
it. On the Kerry Park solve from 1° and 5% with 1 px of keypoint noise,
counting the points whose stored representation disagrees with the test at the
converged level of 0.77 px:

| Iterations a round | Final level (px) | Decided: dirs, disagree, median (px) | Not decided | Crossing between rounds | Crossing off: disagree, median |
|---|---|---|---|---|---|
| 5 | 1.00 | 225, 74, 1.17 | 8, 155, 0.96 | 658, 434, 2.03 | 116, 1.05 |
| 10 | 0.85 | 185, 34, 1.01 | 8, 151, 0.87 | 537, 303, 1.54 | 115, 1.00 |
| 20 | 0.77 | 155, 1, 0.90 | 1, 155, 0.77 | 222, 17, 0.90 | 136, 0.78 |
| 40 (converges) | 0.77 | 158, 1, 0.90 | 1, 158, 0.77 | 218, 16, 0.90 | 155, 0.77 |
| 60, from 4° and 20% (converges) | 0.77 | 149, 5, 0.89 | 1, 151, 0.77 | 265, 53, 0.96 | 176, 0.84 |

"Not decided" is the same solve with the decision skipped, and the crossing
between rounds is the rejected alternative of "Why inverse depth". Deciding is
closer to the converged answer at every budget. The solve's own `ρ` makes few
points directions, because noise puts the best `ρ` of a far point above zero
about half the time, so leaving it undecided keeps nearly every far point
finite. The 20-iteration run does not meet the convergence test, though its
level has already settled at the converged one, so a rule that decided only
after convergence would leave it undecided. No budget brings back the mass
demotion of the crossing between rounds: at five iterations the decision makes
225 of the 886 points directions, against 658, and the median residual is
1.17 px, against 2.03.

**At the default budget an unconverged end is rare, and its level is the
converged one.** Every caller runs 60 iterations a round. Measured through the
reconstruction-level adjustment, which is what `sfm xform --bundle-adjust` and
the viewer's Bundle Adjust run, with each camera's focal and distortion released
where its model admits them, over thirteen inputs -- the incremental and global
solves of `seoul_bull_sculpture`, `seattle_backyard`, `kerry_park` and
`dino_dog_toy` converted to embedded patches and a spline camera
(`SFMTOOL_FISHEYE` for Kerry Park, `SFMTOOL_PINHOLE` for the others), the Kerry
Park global solve after `--find-points-at-infinity`, the Kerry Park ground
truth, the seoul bull
ground truth, the Kerry Park solve and the seoul bull `sift_files` solve -- each
clean and from poses perturbed by 0.2° and 1%, 1° and 5%, and 4° and 20% of the
camera extents with 1 px of keypoint noise: 48 of the 52 solves run (four from
4° and 20% exit degenerate), 6 of those end with an unconverged final round,
none of them from a clean start, and in 5 more an earlier round stops on its
budget and the final round converges. Each unconverged end is compared with the
same input solved at 3,000 iterations a round, where every round converges:

| Input | Start | Level, decided / converged (px) | Directions, decided / converged | Disagree | Disagree at the converged level |
|---|---|---|---|---|---|
| seoul bull global solve | 0.2° and 1% | 0.823 / 0.825 | 0 / 3 | 3 | 3 |
| dino incremental solve | 0.2° and 1% | 1.082 / 1.087 | 1 / 5 | 6 | 6 |
| Kerry Park global solve, points at infinity found | 1° and 5% | 0.733 / 0.733 | 735 / 737 | 2 | 2 |
| Kerry Park global solve, points at infinity found | 4° and 20% | 0.311 / 0.311 | 532 / 532 | 0 | 0 |
| seoul bull `sift_files` solve | 4° and 20% | 0.693 / 0.705 | 34 / 34 | 6 | 6 |
| seoul bull ground truth | 4° and 20% | 0.672 / 0.670 | 5 / 15 | 12 | 12 |

"Disagree" counts the free points whose stored representation differs from the
converged solve's, and the last column the same count when the unconverged state
is decided at the converged solve's level. The levels agree to within 2%, and
deciding at the converged level changes no point, so where the two answers
differ it is because the poses and points differ, not the level. The three from
4° and 20% are solves that failed: their final rounds keep 476 of 4,172, 732 of
3,007 and 552 of 1,277 observations, and their 3,000-iteration references
failed too (median residuals of 48.2 px on the seoul bull ground truth,
165 px on the seoul bull `sift_files` solve and 12.5 px on the Kerry Park
global solve with points at infinity), so those three rows compare two failed
states.

**`converged` reports the iteration budget, not success.** It says whether the
final round met its convergence test, and a solve that has collapsed meets it as
readily as one that has recovered. Of the 48 solves above, 8 more report
`converged: true` from a collapsed state, and their points are decided all the
same:

| Input | Start | Final round keeps | Median residual, crossing on (px) | Directions | Crossing off (px) |
|---|---|---|---|---|---|
| seoul bull incremental solve | 4° and 20% | 20 of 3,258 | 106.7 | 2 | 89.8, 227 points deleted |
| seoul bull global solve | 1° and 5% | 63 of 2,921 | 92.1 | 13 | 50.6, 243 points deleted |
| dino incremental solve | 1° and 5% | 587 of 85,805 | 322.7 | 0 | 249.9, 1,139 points deleted |
| dino global solve | 1° and 5% | 100 of 81,324 | 311.1 | 0 | degenerate exit |
| seattle incremental solve | 4° and 20% | 2,257 of 14,448 | 18.1 | 77 | 0.95, 205 points deleted |
| seattle global solve | 4° and 20% | 909 of 14,386 | 27.8 | 48 | 2.31, 233 points deleted |
| Kerry Park global solve | 4° and 20% | 724 of 2,797 | 9.2 | 163 | 51.3 |
| Kerry Park ground truth | 4° and 20% | 640 of 3,767 | 14.8 | 22 | 19.3 |

The Kerry Park incremental solve from 1° and 5% is borderline (30 of 70 kept,
3.69 px; 5.39 px off). A collapse shows in the final round's kept count and the
median residual, not in `converged`. It is not specific to the crossing: with
the crossing off the same starts collapse in six of the eight, and reach 0.95
and 2.31 px only on seattle, in a draw the five-draw table under "Rough starts
with the lenses released" below shows to be the one of five where that happens.
`FreePointDecision` carries `observation_count`, the finite observations the
level was measured over, which a caller can set against the number it supplied;
stating the kept fraction in the progress line, so a collapse is visible where
the decision is reported, is a possible follow-up.

Growth's adjustments end converged in every case measured, all 49 of them, on
the tracks of the seoul bull, seattle and dino incremental solves, each camera
of the Kerry Park global solve, the Kerry Park solve and the first camera of the
Kerry Park ground truth. `rotation_init` validates no rotation
edge on any of these inputs, which have no far field; on the binding tests'
synthetic far-field scene its finishing adjustment converges for each of four
seeds.

**Remedies, measured where budgets are short.** An unconverged end changes the
answer only at budgets well under the default. From 1° and 5% with 1 px of
keypoint noise, at 5, 10 and 20 iterations a round, each candidate rule is
compared with the converged solve of the same input, counting disagreements as
above:

| Input | Iterations | Level / converged level | As decided | At the converged level | At the previous round's level | At a robust level | No demotion | One more final round | Not decided |
|---|---|---|---|---|---|---|---|---|---|
| Kerry Park solve | 5 | 1.30 | 106 | 91 | 236 | 104 | 161 | 87 | 165 |
| Kerry Park solve | 10 | 1.10 | 78 | 81 | 159 | 94 | 161 | 70 | 163 |
| Kerry Park solve | 20 | 1.00 | 14 | 14 | 11 | 79 | 161 | 13 | 160 |
| Kerry Park global solve | 5 | 1.51 | 192 | 160 | 356 | 160 | 137 | 181 | 150 |
| Kerry Park global solve | 10 | 1.18 | 121 | 114 | 168 | 115 | 137 | 115 | 145 |
| Kerry Park global solve | 20 | 1.01 | 44 | 42 | 52 | 65 | 137 | 27 | 135 |
| Kerry Park ground truth | 5 | 1.56 | 10 | 4 | 20 | 5 | 15 | 19 | 26 |
| Kerry Park ground truth | 10 | 1.81 | 12 | 2 | 18 | 5 | 15 | 14 | 25 |
| Kerry Park ground truth | 20 | 1.18 | 2 | 0 | 15 | 0 | 15 | 13 | 25 |
| seoul bull `sift_files` solve | 5 | 1.46 | 9 | 2 | 25 | 2 | 0 | 3 | 0 |
| seoul bull `sift_files` solve | 10 | 1.25 | 2 | 1 | 16 | 1 | 0 | 2 | 0 |
| seoul bull `sift_files` solve | 20 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Total | | | 590 | 511 | 1,076 | 630 | 939 | 544 | 994 |

"As decided" is the rule above. "At the converged level" decides the same state
at a level no solve can know before it converges, and shows how much of the
error the level accounts for; it is not a bound, and a rule that reads another
level does better than it in a few rows. "At the previous round's level" decides
at the level measured at the end of the second round, "at a robust level" at
1.4826 times the median absolute per-axis residual over the observations the
final round kept, "no demotion" keeps a position every free point handed in as a
position, "one more final round" runs the last schedule round again from where
the solve stopped, at the same budget, and decides there, and "not decided"
stores the solve's own `ρ`. What the table shows:

- **The level is the smaller part of the error.** Deciding at the converged
  level takes the total from 590 disagreements to 511. Those 511 come from
  deciding at poses and points that are not yet the converged ones, which no
  level corrects.
- **The previous round's level is higher**, its trim being wider and its poses
  rougher, and makes more directions: 1,076.
- **A robust level is lower than the converged one even where the level has
  settled** (0.59 to 0.61 px on the two Kerry Park inputs at 20 iterations,
  whose levels are within 1% of the converged 0.77, though the solves have not
  converged): the residuals have heavier tails than a Gaussian, whose spread
  the median-based estimate assumes. It makes far points finite that the
  converged solve stores as directions, 79 against 14 on the Kerry Park solve at
  20 iterations.
- **Not demoting is right only where the converged solve makes no direction.**
  It agrees on the seoul bull `sift_files` solve. On the two Kerry Park solves
  it leaves every one of the 161 and 137 converged directions finite at every
  budget, and on the Kerry Park ground truth, whose 9 bearings are handed in as
  directions, it leaves finite the 15 points the converged solve demotes. It
  is the asymmetric guard the point-or-bearing work measured and rejected, and
  it fails here for the same reason.
- **One more final round** removes 46 disagreements in all, at 0.01 to 0.30 s,
  converges in 1 of the 12 runs, and is worse than deciding on the Kerry Park
  ground truth. On the
  seoul bull global solve from 1° and 5%, which collapses to between 63 and 92
  kept observations at these budgets, it raises the disagreements from 0 to 3 to
  between 86 and 117, and at the default budget, on the Kerry Park global solve
  with points at infinity from 4° and 20%, from 0 to 441. A run to convergence
  is what it approximates, and that is a larger `max_iters`, which is the
  caller's to choose.

So the decision is read at the final round's measured level whether or not that
round converged, `converged: false` and the progress line saying when it did
not. At the default budget the level of an unconverged end is the converged
level; at budgets short enough that it is not, what remains is the unconverged
state, which only more iterations change.

**Rough starts with the lenses released.** The crossing changes which rough
starts the reconstruction-level adjustment recovers from, in both directions,
and with every lens released neither setting recovers reliably from 4° and 20%.
Over five perturbation draws each, the median residual after the solve with the
crossing on and off:

| Input | Start | Crossing on (px) | Crossing off (px) |
|---|---|---|---|
| dino global solve | 0.2° and 1% | 1.62 to 2.71 | 5.20 to 32.98 |
| Kerry Park global solve | 1° and 5% | 0.89, 2.40, 2.13, 1.41, 2.13 | 3.80, 0.94, 1.07, 0.78, 3.39 |
| Kerry Park global solve | 4° and 20% | 3.18 to 16.03 | 28.20 to 151.61 |
| Kerry Park ground truth | 4° and 20% | 14.83 to 58.12, lower in four draws | 19.34 to 92.67 |
| seattle incremental and global solves | 4° and 20% | 18.06 and 27.81 in the first draw; 19.16 to 120.48 in the others | 0.95 and 2.31 in the first draw; 26.19 to 70.12 and one degenerate exit in the others |
| seoul bull incremental solve | 1° and 5% | 0.81 to 1.10 in three draws; 22.3 and 22.5 in two | 0.82 in three draws; 20.3 and 32.5 in two |

With the lenses held, the first seattle draw recovers with the crossing on as
well (1.11 and 0.99 px, against 0.93 and 0.90 off).

**What the crossing left at a converged end, inverse depth does not.** From 4°
and 20% the Kerry Park result disagrees with the test at the converged level on
5 points at 60 iterations and on 3 at 200 (both converge, at the level of
0.77 px), where the crossing between rounds left 53 and 35. Read at the
result's own measured level, which is what `sfm analyze --depth-reliability`
lists, the counts are 35 and 50: that level is the stored measure over every
observation, and the 13 observations over 4 px, which the adjustment's last
trim left out and the stored measure's gate passes, raise it to 0.86 and
0.89 px, where the decision read the 0.77 px of the observations it solved on.

**Cost is the solve's own.** The crossing changes how many iterations the
rounds take, not what an iteration costs. Seconds, best of five runs alternating
crossing off and on (three for the 1° and 5% starts and for dino growth, seven
for `rotation_init`), on the inputs above:

| Caller | Input | Off (s) | On (s) | On / off |
|---|---|---|---|---|
| reconstruction-level adjustment | seoul bull incremental solve | 0.169 | 0.151 | 0.89 |
| reconstruction-level adjustment | seoul bull global solve | 0.292 | 0.471 | 1.62 |
| reconstruction-level adjustment | seattle incremental solve | 0.904 | 0.746 | 0.83 |
| reconstruction-level adjustment | seattle global solve | 0.668 | 0.456 | 0.68 |
| reconstruction-level adjustment | Kerry Park global solve | 0.225 | 0.120 | 0.53 |
| reconstruction-level adjustment | Kerry Park global solve, points at infinity found | 0.180 | 0.255 | 1.42 |
| reconstruction-level adjustment | dino incremental solve | 11.903 | 8.852 | 0.74 |
| reconstruction-level adjustment | dino global solve | 9.965 | 5.480 | 0.55 |
| reconstruction-level adjustment | Kerry Park ground truth | 0.400 | 0.226 | 0.57 |
| reconstruction-level adjustment | Kerry Park solve | 0.110 | 0.053 | 0.48 |
| reconstruction-level adjustment | seoul bull `sift_files` solve | 0.042 | 0.030 | 0.72 |
| reconstruction-level adjustment | seoul bull ground truth | 0.033 | 0.021 | 0.64 |
| reconstruction-level adjustment, from 1° and 5% | Kerry Park solve | 0.516 | 0.394 | 0.77 |
| reconstruction-level adjustment, from 1° and 5% | Kerry Park ground truth | 1.216 | 1.397 | 1.15 |
| reconstruction-level adjustment, from 1° and 5% | Kerry Park global solve | 0.156 | 0.418 | 2.68 |
| reconstruction-level adjustment, from 1° and 5% | seattle global solve | 1.516 | 1.833 | 1.21 |
| `grow_reconstruction` | seoul bull incremental solve | 0.339 | 0.295 | 0.87 |
| `grow_reconstruction` | seattle incremental solve | 2.445 | 2.103 | 0.86 |
| `grow_reconstruction` | Kerry Park solve | 0.745 | 0.515 | 0.69 |
| `grow_reconstruction` | Kerry Park ground truth, first camera | 0.418 | 0.180 | 0.43 |
| `grow_reconstruction` | dino incremental solve | 46.658 | 27.771 | 0.60 |
| `rotation_init` | synthetic far-field scene, four seeds | 0.059 to 0.432 | 0.056 to 0.430 | 0.92 to 1.03 |

`rotation_init` has no opt-out, so its "off" column was timed through a switch
built for the measurement and not kept.

Timed round by round, an iteration costs the same in either parametrisation:
3.4 ms off and 3.6 ms on for the seoul bull global solve, and 1.198 s and
1.209 s for the 11 iterations of the first round of the dino global solve. The
re-estimation under the crossing costs 0.08 to 0.09 s a round on the dino global
solve, against 0.03 off, and the decision 0.015 s. The slower runs take more
iterations: the seoul bull global solve takes 59, 16 and 17 iterations a round
off and 60, 60 and 17 on; the Kerry Park global solve with points at infinity,
13, 11 and 11 off and 28, 6 and 9 on; the seattle global solve from 1° and 5%,
39, 11 and 36 off and 53, 22 and 24 on; the Kerry Park ground truth from 1°
and 5%, 29, 25 and 37 off and 25, 33 and 55 on. The dino global solve saves the same
way, with 11, 32 and 34 off and 11, 6 and 26 on. The Kerry Park global solve
from 1° and 5% takes longer because its second round runs the whole budget and
the solve recovers, to 0.886 px, where with the crossing off that round stops
after 15 iterations and the solve ends at 3.797 px. The time is spent on the
path the solve takes rather than on any step the crossing adds, so the
crossing's own work holds nothing to remove.

**Inverse depth is the crossing, not a second switch.** The two halves are not
useful apart. Inverse depth without the decision stores the solve's own `ρ`,
which leaves almost every far point finite (the "Not decided" column above: 151
to 158 disagreements at the converged level where the decision leaves 1 to 74),
and a decision on a point carried through the solve as a position or a
direction reads the score that representation has pulled toward itself (the
synthetic measurement above). With the switch off every free point keeps the
representation it is handed in, which is what a caller that opts out relies
on.

**The level is measured, not the loss scale.** `loss_scale` is a schedule
constant, chosen before anything is measured, and one schedule runs over
captures whose noise differs fivefold: the levels the decision reads on the
clean inputs run from 0.19 px on the Kerry Park solve to 1.05 px on
`dino_dog_toy`. The level comes from the final round's kept observations, the
residuals the solve has just minimised, whose trim capped every residual it
kept at 4 px by default, which bounds what a gross outlier the gate passes can
add.

#### Measured against the crossing between rounds

The default schedule over the four inputs of the point-or-bearing work, clean
and degraded (poses perturbed by 0.2° and 1% of the camera extents, 1 px of
keypoint noise, both, and 1° and 5% with 1 px), and the same 1° and 5% start
with five iterations a round, every free point crossing and the focal held.
"Disagree" counts the points whose stored representation the test at the
result's own measured noise disagrees with; "dirs" are the directions in and
out. Every default-iteration run converges.

| Input | Start | Between rounds: dirs, disagree, median (px) | Inverse depth: dirs, disagree, median (px) |
|---|---|---|---|
| Kerry Park ground truth | clean | 9 → 9, 0, 0.167 | 9 → 3, 0, 0.165 |
| Kerry Park ground truth | 1 px keypoints | 9 → 12, 0, 1.016 | 9 → 12, 0, 1.022 |
| Kerry Park ground truth | 0.2° and 1%, 1 px | 9 → 12, 1, 1.018 | 9 → 12, 1, 1.021 |
| Kerry Park ground truth | 1° and 5%, 1 px | 9 → 12, 1, 1.018 | 9 → 12, 1, 1.019 |
| seoul bull ground truth | clean | 14 → 14, 0, 0.256 | 14 → 13, 0, 0.254 |
| seoul bull ground truth | 1 px keypoints | 14 → 14, 0, 0.913 | 14 → 14, 0, 0.913 |
| seoul bull ground truth | 0.2° and 1%, 1 px | 14 → 14, 0, 0.911 | 14 → 14, 0, 0.912 |
| seoul bull ground truth | 1° and 5%, 1 px | 14 → 14, 1, 0.921 | 14 → 14, 0, 0.925 |
| Kerry Park solve | clean | 0 → 0, 0, 0.173 | 0 → 0, 0, 0.173 |
| Kerry Park solve | 1 px keypoints | 0 → 174, 27, 0.863 | 0 → 134, 7, 0.869 |
| Kerry Park solve | 0.2° and 1%, 1 px | 0 → 213, 34, 0.892 | 0 → 161, 2, 0.899 |
| Kerry Park solve | 1° and 5%, 1 px | 0 → 218, 33, 0.897 | 0 → 161, 10, 0.901 |
| seoul bull `sift_files` solve | every start | 0 → 0, 0 | 0 → 0, 0 |
| Kerry Park ground truth | 1° and 5%, 5 iterations | 9 → 43, 78, 3.75 | 9 → 27, 83, 1.39 |
| seoul bull ground truth | 1° and 5%, 5 iterations | 14 → 24, 114, 4.29 | 14 → 13, 87, 1.51 |
| Kerry Park solve | 1° and 5%, 5 iterations | 0 → 658, 58, 2.03 | 0 → 225, 272, 1.17 |
| seoul bull `sift_files` solve | 1° and 5%, 5 iterations | 0 → 75, 228, 6.29 | 0 → 9, 276, 2.81 |

On the converged Kerry Park runs the disagreements fall from 27 to 34 to 2 to
10, and the directions from 174 to 218 to 134 to 161: the crossing between
rounds left marginal far points still settling as directions when the three
rounds ended, and inverse depth settles them within the rounds. The
disagreements on the converged degraded runs are read at the result's own
level, a little above the level the decision read; at the decision's level
every converged run in the table agrees with the test on every point. On the
clean Kerry Park ground truth the solve makes 6 of its 9 bearings finite
(156, 157, 176, 268, 269 and 270), where the crossing between rounds makes
none. The six score 9 to 16 at the input, 9 to 16 after the solve that carries
them as directions, and 28 to 34 after the one that carries them in inverse
depth. They are depth
the capture supports. Rebuilt synthetically on the Kerry Park ground truth's
own geometry -- its poses, lenses and finite points, with the observation
patterns of its 9 bearings repeated five times (45 far tracks) and 0.21 px of
noise, over four draws (ten for the tracks truly at infinity) -- tracks that
are truly at infinity are made finite by the inverse-depth solve in 0 of 450
cases (mean score 0.5 to 1.8, against 0.2 to 1.1 at the true poses). Tracks
truly 2,500 m out score about 37.9 to 42.8 at the true poses and 41.3 to 46.1
after the inverse-depth solve, which makes 166 of 180 finite, but about 3.8 to
4.6 when carried as directions: the crossing between rounds makes 2 of 180
finite and the crossing off 0. At 5,000 m 9 of 180 are finite at the true
poses and 26 of 180 after the inverse-depth solve. A lens error moves the
scores of true bearings further in inverse depth than when they are carried as
directions: with the focal 1% off, the mean score of the truly infinite tracks
is about 58 to 66 against about 35 under the crossing between rounds, and 1 of
180 is made finite; with it 0.3% off, the mean score is about 7 to 9 against
about 4, and none is made finite. The five-iteration
disagreements are read at the result's own level, which unconverged poses put
at 3.1 to 12 px on the three inputs other than the Kerry Park solve, a level at
which the test calls most points bearings; the decision's level, read over the
observations the final round kept, is 1.0 to 1.3 px there. Those runs leave 28
to 315 points unscored, each in the representation it was handed in with. No input produced a
`NaN` point.

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
  makes no point a direction. (The crossing between rounds made at least 25 of
  them directions at its first re-estimation with three iterations a round.)
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
- Per-observation camera models: an observation's camera is its image's.
- Optimizing distortion beyond the `k1` of `SIMPLE_RADIAL_FISHEYE` and the
  spline of `SFMTOOL_FISHEYE` / `SFMTOOL_PINHOLE`, or the principal point;
  `opt_f`/`opt_k1`/`opt_bspline` cover each camera's focal and those two radial
  releases only.
- Gauge fixing and covariance estimation. The kernel holds no pose fixed to
  set the gauge and reports no uncertainty. A caller that wants a gauge states
  it through the point constraints (held points, or ranged points, which fix
  the scale; see "Point constraints") or aligns the result afterwards.
- Replacing the production solvers (`sfm solve` wraps COLMAP/GLOMAP). The
  kernel runs inside reconstruction growth and rotation initialization, and
  through the reconstruction-level adjustment it backs the viewer's Bundle
  Adjust command, the `bundle_adjust` MCP tool and `sfm xform --bundle-adjust`
  on spline cameras.
