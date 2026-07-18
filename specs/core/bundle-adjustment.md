# Staged bundle adjustment (shared camera)

**Status:** Implemented —
`crates/sfmtool-core/src/geometry/bundle_adjust.rs` (`bundle_adjust`,
`BaSchedule`, `BundleAdjustment`; tests in `bundle_adjust/tests.rs`), PyO3
binding in `crates/sfmtool-py/src/geometry/bundle_adjust.rs`
(`sfmtool._sfmtool.geometry.bundle_adjust`), Python tests in
`tests/rust_bindings/test_bundle_adjust_rust_bindings.py`.

## Purpose

The staged robust bundle adjustment used by the cluster pinhole bootstrap
(`specs/core/cluster-pinhole-bootstrap.md`,
`scripts/exp_fast_pinhole.py` / `scripts/exp_pinhole_bootstrap.py`): given
images sharing one camera model, camera poses, world points, and pixel
observations tying them together, jointly refine the poses and points (and
optionally the shared focal length) by minimizing robust pixel reprojection
error over a trim schedule with inter-round retriangulation.

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
    schedule: &[BaSchedule],             // default 50/5 → 12/2 → 4/1
    max_iters: usize,                    // LM iterations per round
    min_track: usize,                    // trim survivors per point (2)
    min_obs: usize,                      // degenerate-exit floor (12)
) -> BundleAdjustment;                   // { focal, residual_norms }
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
  `opt_f`, the shared focal `f ← f + δf`. Focal optimization requires
  `SIMPLE_PINHOLE` (single focal, no distortion), where
  `∂(u, v)/∂f = ((u − cx)/f, (v − cy)/f)` exactly; the binding rejects
  `opt_f` for other models loudly, and the core silently degrades it to a
  fixed-focal solve (never a half-modeled focal DOF).
- **Jacobian.** The projection block `∂(u, v)/∂p_cam` — analytic from
  `CameraIntrinsics::ray_to_pixel_with_jacobian` for the perspective
  family, a central difference of `ray_to_pixel` for fisheye /
  equirectangular models that have no analytic form — composed with
  `−[R·X]ₓ` (rotation), `I₃` (translation), and `R` (point) blocks,
  exactly as in `pose_refine.rs` (including the fallback). An observation
  whose point is behind the camera / outside the model domain contributes
  residual `(1e6, 0)` with a zero Jacobian row — penalized, never
  steering.
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
  (`[f? | 6·n_im]`, dense) is solved by LU; point updates back-substitute.
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
    opt_f=False,               # requires SIMPLE_PINHOLE
    schedule=[(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)],
    max_iters=60,
    min_track=2,
    min_obs=12,
) -> dict                      # focal, quaternions_wxyz (n_img, 4),
                               # translations (n_img, 3), points (n_pt, 3),
                               # residual_norms (n_obs,)
```

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
- **Memory order**: Fortran-ordered inputs to the binding produce the same
  result as C-ordered ones (guards the `to_contiguous!` zero-copy path
  against silent transposition).
- **Binding behavior**: the Python binding reproduces the kernel's
  behavior on analogous synthetic scenes (`tests/rust_bindings/`).

## Points at infinity

**Status:** Specified — implementation in progress.

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
  or an all-`false` mask reproduces the finite-only kernel bit for bit.
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
  translation. The `min_obs` degenerate-exit floor counts **finite-point
  survivors only**, and an image whose surviving observations are all
  directions has its translation frozen for that round (its rotation
  still updates); otherwise the reduced camera system would carry a
  zero-curvature translation block.

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

- **Regression**: an all-`false` mask and an absent mask both reproduce
  the finite-only kernel's output bit for bit on the existing synthetic
  scenes.
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
- **Memory order and binding parity** as for the finite kernel.

## Non-goals

- Per-image or per-observation camera models — one shared
  `CameraIntrinsics`.
- Optimizing distortion or principal point; `opt_f` covers the single
  shared focal only.
- Gauge fixing, covariance estimation, or constraint handling — callers
  own the gauge (the bootstrap's evaluation aligns by similarity anyway).
- Replacing the production solvers (`sfm solve` wraps COLMAP/GLOMAP); this
  kernel serves the bootstrap experiments and whatever grows out of them.
