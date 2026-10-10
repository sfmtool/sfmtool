# Rotation-Locked Resection

## Purpose

Solve a camera's translation against known world points when its rotation
is already known. With the rotation fixed the problem is linear in the
three translation components, which makes the solve stable exactly where
full 6-DOF resection is fragile: low-parallax observations constrain a
translation firmly while leaving a joint rotation–translation solve free
to trade the two against each other. The kernel does not care where the
rotation came from. Its one in-repo caller is the far-field rotation
skeleton in
[rotation_init.rs](../../../crates/sfmtool-core/src/geometry/rotation_init.rs),
which resects each camera's position after fixing its rotation; the
Python binding exposes the same function for any other source of
rotation.

## Mechanism

Inputs: `CameraIntrinsics`, world-to-camera rotation `R`, world points
`X_k` (`f64 [n, 3]`), observed pixels `uv_k` (`f64 [n, 2]`),
`max_error_px` (trim gate), `min_inliers`. The core function takes the
last two as required arguments and carries no defaults: `8.0` and `10`
are the Python binding's signature defaults (see
[Interface](#interface)), and the one in-repo Rust caller, the far-field
rotation skeleton, passes its own `RESECT_MAX_ERROR_PX` /
`RESECT_MIN_INLIERS`.

The core function returns `None` before any solve when `uv` and `points`
differ in length, or when there are fewer than `max(min_inliers, 1)`
observations.

Each observation's ray `r_k = pixel_to_ray(uv_k)` (normalized to unit
length, camera frame) must be parallel to `R·X_k + t`:

```
[r_k]ₓ · (R·X_k + t) = 0    →    [r_k]ₓ · t = −[r_k]ₓ · R·X_k
```

Three linear rows per observation (rank 2). An observation whose
`pixel_to_ray` is non-finite or has length below `1e-12`, or whose
rotated world point `R·X_k` is non-finite, is excluded from every
solve and every kept set; if fewer than `min_inliers` observations
remain after that exclusion, the resection fails before the first
round. The solve is trimmed iteratively reweighted least squares:

1. Least-squares solve over the current observation set (all valid
   observations, initially).
2. Reproject: keep observations in front of the camera with pixel
   residual below `max_error_px`.
3. Repeat 3 rounds or until the kept set is stable. A round that keeps
   fewer than 2 observations fails the resection, since two rays are the
   fewest that determine `t` and the next solve would be underdetermined.
4. When the 3 rounds end with the last gate still changing the set, solve
   once more over the final kept set and gate once more at that solve. The
   returned `t` is therefore always the least-squares fit over the returned
   kept set, whether or not the set stabilised.
5. Fewer than `min_inliers` members of the final kept set within the gate at
   the returned `t` fails the resection. When the set stabilised that is the
   whole kept set. An earlier round may keep fewer: the next solve runs over
   that round's survivors and can re-admit observations, so the floor is read
   on the set the trim ends with, not on the way there.

Reading the floor on every round dropped resections that converge well. On
`KerryPark480`'s seed with its mover members dropped, image 31 failed on an
intermediate round although its final round kept 31 observations; read on
the final set it resects with 31.

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

Output: `t`, the surviving-observation mask (the final kept set, which `t`
is the fit over), and pixel residual norms. When the set did not stabilise, a
member of the mask can lie outside the gate at `t`; its residual says so.
All three outputs are per **input** observation and length `n`: a
non-survivor keeps the residual it scored at the final translation, and
an observation the camera cannot image — behind it, outside the model's
domain, or excluded above for a non-finite or zero-length ray or a
non-finite point — reports `INVALID_RESIDUAL` (`1e6`)
rather than a real number. Zipping `residual_norms` against the inlier
subset instead of the inputs misaligns it.

## Interface

The kernel lives in
[resect_translation.rs](../../../crates/sfmtool-core/src/geometry/resect_translation.rs):

```rust
pub struct TranslationResection {
    pub translation: Vector3<f64>,   // world-to-camera t
    pub inliers: Vec<bool>,          // per input observation
    pub residual_norms: Vec<f64>,    // per input observation
}

pub fn resect_translation(
    cam: &CameraIntrinsics,
    rotation: &UnitQuaternion<f64>,
    points: &[[f64; 3]],
    uv: &[[f64; 2]],
    max_error_px: f64,
    min_inliers: usize,
) -> Option<TranslationResection>;
```

It is bound as `sfmtool.geometry.resect_translation` by
[resect_translation.rs](../../../crates/sfmtool-py/src/geometry/resect_translation.rs),
which is where the two trim defaults live:

```python
resect_translation(camera, rotation_wxyz, points, uv,
                   max_error_px=8.0, min_inliers=10)
    -> {"translation": (3,), "inliers": (n,) bool,
        "residual_norms": (n,)} | None
```

The binding raises `ValueError` when `points` is not `(N, 3)`, `uv` is
not `(N, 2)`, or the two differ in length, so the core's
length-mismatch `None` never reaches Python.

## Callers

The far-field rotation skeleton in
[rotation_init.rs](../../../crates/sfmtool-core/src/geometry/rotation_init.rs)
keeps an image whose resection fails among its candidates, so the image is
tried again after the next round of growth adds structure. Its growth loop
calls this kernel directly, so the round-survivor minimum, the extra solve
over a set that did not stabilise and the floor read on the final kept set
apply to the skeleton.

## Testing requirements

- Exact recovery on noiseless synthetic data, pinhole and fisheye.
- Contamination: planted outliers beyond the gate are trimmed and do not
  bias `t`; the returned mask identifies them.
- Behind-camera points are excluded by the cheirality check, under both
  readings: a fisheye camera keeps its past-90° observations (a
  half-space gate would drop them) and still rejects the antipodal
  reflection along each ray. A perspective camera evaluates the same
  half-space expression it always did.
- Failure path: fewer than `min_inliers` consistent observations
  returns `None` (core and binding).
- A trim round that dips under `min_inliers` does not fail a resection
  whose final kept set clears it. A round that keeps one observation fails
  the resection even under a floor of 1; a round that keeps two continues.
  When the rounds run out before the set stabilises, the returned
  translation is the least-squares fit over the returned kept set. `uv` and
  `points` of different
  lengths return `None` from the core and raise `ValueError` from the
  binding.
- Degenerate ray bundles (all rays near-parallel) still return the
  least-squares `t` — conditioning is the caller's concern, correctness
  of the normal equations is this kernel's.
- Binding parity and memory-order guards as elsewhere.

## Non-goals

- Rotation refinement — `refine_absolute_pose` exists for joint updates.
- RANSAC over correspondence hypotheses; the trimmed IRLS assumes
  correspondences are largely correct (cluster tracks), with the gate
  handling stragglers.
