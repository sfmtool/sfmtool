# Rotation-Locked Resection

**Status:** Implemented (2026-07-18) —
`crates/sfmtool-core/src/geometry/resect_translation.rs` (kernel, tests in
`resect_translation/tests.rs`), PyO3 binding in
`crates/sfmtool-py/src/geometry/resect_translation.rs`
(`sfmtool._sfmtool.geometry.resect_translation`), Python tests in
`tests/rust_bindings/test_resect_translation_rust_bindings.py`. Joins the
geometry module beside `estimate_absolute_pose` / `refine_absolute_pose`.

## Purpose

Solve a camera's translation against known world points when its rotation
is already known. With the rotation fixed the problem is linear in the
three translation components, which makes the solve stable exactly where
full 6-DOF resection is fragile: low-parallax observations constrain a
translation firmly while leaving a joint rotation–translation solve free
to trade the two against each other. Callers with a rotation from any
source — a far-field rotation skeleton, a rig calibration, an external
attitude — resect position only.

## Mechanism

Inputs: `CameraIntrinsics`, world-to-camera rotation `R`, world points
`X_k` (`f64 [n, 3]`), observed pixels `uv_k` (`f64 [n, 2]`),
`max_error_px` (trim gate, default `8.0`), `min_inliers` (default `10`).

Each observation's ray `r_k = pixel_to_ray(uv_k)` (unit, camera frame)
must be parallel to `R·X_k + t`:

```
[r_k]ₓ · (R·X_k + t) = 0    →    [r_k]ₓ · t = −[r_k]ₓ · R·X_k
```

Three linear rows per observation (rank 2). The solve is trimmed
iteratively reweighted least squares:

1. Least-squares solve over the current observation set (all, initially).
2. Reproject: keep observations in front of the camera with pixel
   residual below `max_error_px`.
3. Repeat 3 rounds or until the kept set is stable. Fewer than
   `min_inliers` survivors at any round fails the resection.

Working in ray space makes the equations camera-model-agnostic: fisheye
and equirectangular observations resect through the same rows,
`pixel_to_ray` absorbing the model. The residual gate is evaluated in
pixels through `ray_to_pixel`.

The rows are sign-blind — `[r_k]ₓ·(R·X_k + t)` vanishes for `−r_k` too,
so the equations cannot tell a point from its reflection through the
camera centre — which makes step 2's in-front test the carrier of the
chirality, and it is therefore model-dependent:

- **Perspective family:** the half-space `(R·X_k + t)_z < 0` (canonical
  camera, `−Z` forward), which is also exactly that family's projection
  domain.
- **`needs_ray_path` models** (fisheye, equirectangular): positive range
  along the observed ray, `r_k·(R·X_k + t) > 0`. Such a camera images
  past 90° off axis, and the half-space would reject that whole
  periphery — precisely the population this kernel's model-agnosticism
  is about — while the range test still rejects the antipodal
  reflection, which is the one thing the sign-blind rows need the gate
  for.

Output: `t`, the surviving-observation mask, and the survivors' pixel
residual norms.

## Binding

```python
resect_translation(camera, rotation_wxyz, points, uv,
                   max_error_px=8.0, min_inliers=10)
    -> {"translation": (3,), "inliers": (n,) bool,
        "residual_norms": (n,)} | None
```

## Testing requirements

- Exact recovery on noiseless synthetic data, pinhole and fisheye.
- Contamination: planted outliers beyond the gate are trimmed and do not
  bias `t`; the returned mask identifies them.
- Behind-camera points are excluded by the cheirality check, under both
  readings: a fisheye camera keeps its past-90° observations (a
  half-space gate would drop them) and still rejects the antipodal
  reflection along each ray. A perspective camera evaluates the same
  half-space expression it always did.
- Failure path: fewer than `min_inliers` consistent observations returns
  `None` (binding) / failure (core).
- Degenerate ray bundles (all rays near-parallel) still return the
  least-squares `t` — conditioning is the caller's concern, correctness
  of the normal equations is this kernel's.
- Binding parity and memory-order guards as elsewhere.

## Non-goals

- Rotation refinement — `refine_absolute_pose` exists for joint updates.
- RANSAC over correspondence hypotheses; the trimmed IRLS assumes
  correspondences are largely correct (cluster tracks), with the gate
  handling stragglers.
