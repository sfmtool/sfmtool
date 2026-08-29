# Staged bundle adjustment (shared camera)

**Status:** Implemented —
`crates/sfmtool-core/src/geometry/bundle_adjust.rs` (`bundle_adjust`,
`BaSchedule`, `BundleAdjustment`; tests in `bundle_adjust/tests.rs`), PyO3
binding in `crates/sfmtool-py/src/geometry/bundle_adjust.rs`
(`sfmtool._sfmtool.geometry.bundle_adjust`), Python tests in
`tests/rust_bindings/test_bundle_adjust_rust_bindings.py`.

## Purpose

The staged robust bundle adjustment used by the cluster pinhole bootstrap
(`specs/core/geometry/cluster-pinhole-bootstrap.md`,
`scripts/exp_fast_pinhole.py` / `scripts/exp_pinhole_bootstrap.py`): given
images sharing one camera model, camera poses, world points, and pixel
observations tying them together, jointly refine the poses and points (and
optionally the shared focal length and the shared distortion release — a
radial coefficient or a spline) by minimizing
robust pixel reprojection error over a trim schedule with inter-round retriangulation.

This is the optimizer that the trimmed pose-only refinement
(`crates/sfmtool-core/src/geometry/pose_refine.rs`) is the single-pose
special case of. It replaces the experiment scripts'
`scipy.optimize.least_squares` BA, whose Python-side residual and sparsity
handling dominated the bootstrap's wall-clock.

## Definitions

- `n_img` **images** sharing one `CameraIntrinsics`, each with a
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

```rust
pub struct BaSchedule {
    pub trim_px: f64,     // pre-round trim threshold on the residual norm
    pub loss_scale: f64,  // soft-L1 scale for the round's solve, px
}

pub fn bundle_adjust(
    cam: &CameraIntrinsics,          // shared model; carries the initial focal
    quats: &mut [UnitQuaternion<f64>],   // n_img, world-to-camera
    trans: &mut [Vector3<f64>],          // n_img
    points: &mut [[f64; 3]],             // n_pt (NaN allowed)
    uv: &[[f64; 2]],                     // n_obs
    obs_img: &[u32],                     // n_obs
    obs_pt: &[u32],                      // n_obs
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],             // default 50/5 → 12/2 → 4/1
    max_iters: usize,                    // LM iterations per round
    min_track: usize,                    // trim survivors per point (2)
    min_obs: usize,                      // degenerate-exit floor (12)
) -> BundleAdjustment;                   // { focal, k1, bspline, residual_norms }
```

Per schedule round, mirroring the experiment scripts exactly:

1. **Retriangulate (rounds after the first).** Rebuild *every* point from
   *all* supplied observations at the current poses: world rays
   `R_iᵀ · pixel_to_ray(uv)` and centers `−R_iᵀ t_i` per observation,
   grouped by point, through
   [`reconstruction::triangulation::triangulate_batch`]. A track with fewer
   than 2 observations becomes `NaN`; a point with no observations at all
   becomes `NaN` too (the callers refill from their full observation set —
   the "refill after BA" rule of the bootstrap spec). Re-admission is the
   point: observations a bad init lost re-enter once the refined cameras
   explain them.
2. **Trim.** Keep observations with residual norm `< trim_px`, in-front
   depth `> 1e-3 · f` (canonical depth is `−z_cam`), and a finite point;
   then drop observations of points with fewer than `min_track` survivors.
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
  (`R ← exp(δθ)·R`, `t ← t + δt`); per touched point `X ← X + δX`; when
  `opt_f`, the shared focal `f ← f + δf`; when `opt_k1`, the shared
  radial coefficient `k1 ← k1 + δk1`; when `opt_bspline`, the shared
  spline `cᵢ ← cᵢ + δcᵢ`. Focal optimization requires a
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
  `opt_k1` for those loudly, and the core silently degrades to the
  fixed-parameter solve (never a half-modeled DOF). The rung also needs the
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
  The released block is the whole coefficient vector `c₀..c_{N−1}`, shared by
  every image like `f` and `k1`, so the reduced camera system is
  `[f, k1, c₀..c_{N−1} | 6·n_im]` — width `2 + N + 6·n_im`. `opt_k1` and
  `opt_bspline` are mutually exclusive: no model carries both parameters, so
  the binding rejects the combination (checked before the model gates, so the
  caller sees the real reason) and the core degrades each on its own model
  test. Retriangulation and direction re-estimation read the model's Newton
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
  invariant `1 + δ'(d) > 0` anywhere on `[0, d_max]` is rejected and the step
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
  periphery past 90° included; on the spline's held-constant tail past
  `d_max` the basis is evaluated at `d_max`, which is exactly the derivative
  of the held constant. A direction (point at infinity) takes every one of these
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
  inverted directly and the reduced camera system
  (`[f?, k1?, c₀..c_{N−1}? | 6·n_im]`, dense) is solved by LU; point updates
  back-substitute. Unreleased shared slots (and released coefficient slots
  with no observation support) are pinned to an identity row/column with a
  zero gradient entry, which is what keeps that system regular.
  Rejected steps re-damp and re-solve from the same linearization (no
  re-evaluation), with Marquardt scaling `λ·diag(JᵀJ)` for the
  `x_scale="jac"` parameter-scale invariance of the scipy original.
- **Termination.** `max_iters` accepted-step budget per round; stop early
  when accepted steps improve the cost by less than `1e-8` relative TWICE
  in a row (one tiny step is how a traverse of a nearly-flat valley starts
  — the focal release walks −20% through one), or when no damping in a
  bounded ladder (12 ×4 escalations, capped at `λ = 10¹²`) finds a
  downhill step.

## Bindings

```python
bundle_adjust(
    camera,                    # CameraIntrinsics shared by all images (initial f)
    quaternions_wxyz,          # (n_img, 4) world-to-camera (WXYZ)
    translations,              # (n_img, 3)
    points,                    # (n_pt, 3), NaN allowed
    uv,                        # (n_obs, 2)
    obs_image,                 # (n_obs,) uint32
    obs_point,                 # (n_obs,) uint32
    opt_f=False,               # SIMPLE_PINHOLE, EQUIDISTANT_FISHEYE,
                               # SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE
                               # or SFMTOOL_PINHOLE
    opt_k1=False,              # SIMPLE_RADIAL_FISHEYE only
    opt_bspline=False,         # SFMTOOL_FISHEYE or SFMTOOL_PINHOLE,
                               # defined spline; exclusive with opt_k1
    schedule=[(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)],
    max_iters=60,
    min_track=2,
    min_obs=12,
) -> dict                      # focal, k1, bspline_coefficients (n_coeffs,),
                               # quaternions_wxyz (n_img, 4),
                               # translations (n_img, 3), points (n_pt, 3),
                               # residual_norms (n_obs,)
```

`bspline_coefficients` is always present: the coefficients after the solve —
the camera's input ones unless `opt_bspline` — and an empty array for models
that carry no spline, mirroring how `k1` reports `0.0` for models without
one. `opt_bspline` raises for a camera that is neither `SFMTOOL_FISHEYE` nor
`SFMTOOL_PINHOLE`, and for an undefined spline; `opt_k1` together with
`opt_bspline` raises first, so the caller sees the exclusion rather than
whichever model gate happens to fire.

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

## Points at infinity

**Status:** Implemented — same locations as the kernel above. There is one
staged loop, not two: direction handling is a per-point branch inside it, so
with nothing marked the loop *is* the finite-only solve. (It was originally a
second mirrored copy of the whole kernel, entered only when a direction was
marked; the copies were merged once the reduction was shown to hold bit for
bit.)

A point at infinity is a pure direction: its observations depend on the
observing image's rotation and the shared camera model, never on any
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
  distinction, and it is not modified — classification belongs to the
  caller.

### Residuals and derivatives

A direction projects like a point at infinite depth: `uv_pred =
ray_to_pixel(R_i · d)`. The residual is the same pixel difference as a
finite observation — same units, same soft-L1 loss, same trim thresholds.
A direction "in front" satisfies `(R_i · d)_z < 0` (canonical −Z
forward); a behind-camera or out-of-domain direction contributes the
standard `(1e6, 0)` penalized residual with a zero Jacobian row.

- **Parameters.** A direction perturbs in the 2-DOF tangent plane of the
  unit sphere: `d ← normalize(d + B(d) · δ)` with `B(d)` an orthonormal
  basis of `d⊥` rebuilt at each linearization. Its Schur block is 2×2
  where a finite point's is 3×3; the translation Jacobian block is zero;
  the rotation block is `−[R·d]ₓ` composed with the same projection
  Jacobian as finite points, and the `opt_f` derivative applies
  unchanged.
- **Translation observability.** Infinity observations constrain no
  translation, so an image whose surviving observations are all directions
  has its translation frozen for that round (its rotation still updates);
  otherwise the reduced camera system would carry a zero-curvature
  translation block. The `min_obs` degenerate-exit floor is independent of
  that and counts **every trim survivor**, finite and direction alike: it
  measures whether the round retained enough evidence to solve on, and a
  direction constrains the rotations and the shared camera parameters just
  as a finite observation does. A directions-only observation set therefore
  runs at the default floor.

### Staged-loop semantics

- **Trim** treats direction observations exactly like finite ones (pixel
  threshold, `min_track` survivors per point); the in-front check is the
  cheirality test above instead of the depth floor.
- **Re-estimation (rounds after the first).** Where finite points
  retriangulate, a direction re-estimates in closed form as the
  normalized mean of its observations' back-rotated rays
  `R_iᵀ · pixel_to_ray(uv)` at the current rotations. A direction track
  with fewer than 2 observations becomes `NaN`, mirroring finite tracks.

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

## Protected observations

**Status:** Implemented — same locations as the kernel above; an absent or
all-`false` mask reproduces the unprotected behavior bit for bit.

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

- Per-image or per-observation camera models — one shared
  `CameraIntrinsics`.
- Optimizing distortion beyond the single shared `k1` of
  `SIMPLE_RADIAL_FISHEYE` and the shared spline of `SFMTOOL_FISHEYE` /
  `SFMTOOL_PINHOLE`, or the principal point; `opt_f`/`opt_k1`/`opt_bspline`
  cover the shared focal and those two radial releases only.
- Gauge fixing, covariance estimation, or constraint handling — callers
  own the gauge (the bootstrap's evaluation aligns by similarity anyway).
- Replacing the production solvers (`sfm solve` wraps COLMAP/GLOMAP); this
  kernel serves the bootstrap experiments and whatever grows out of them.
