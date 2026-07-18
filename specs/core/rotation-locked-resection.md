# Rotation-Locked Resection

**Status:** Specified — not implemented. Joins the geometry module beside
`estimate_absolute_pose` / `refine_absolute_pose`.

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
2. Reproject: keep observations with canonical depth in front
   (`(R·X_k + t)_z < 0`) and pixel residual below `max_error_px`.
3. Repeat 3 rounds or until the kept set is stable. Fewer than
   `min_inliers` survivors at any round fails the resection.

Working in ray space makes the mechanism camera-model-agnostic: fisheye
and equirectangular observations resect through the same equations,
`pixel_to_ray` absorbing the model. The residual gate is evaluated in
pixels through `ray_to_pixel`.

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
- Behind-camera points are excluded by the cheirality check.
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
